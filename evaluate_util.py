import os 
import sys
import hydra
import json
import gc
from tqdm import tqdm
import torch.nn as nn
import numpy as np 
import math
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import zlib
from collections import defaultdict
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    set_seed, 
    LlavaForConditionalGeneration, 
    AutoProcessor,
    CLIPImageProcessor,
    # MllamaForConditionalGeneration, 
)
from pathlib import Path
from rouge_score import rouge_scorer
from forget import find_all_linear_names
from utils import get_model_identifiers_from_yaml
from data_module import (
    MMDatasetQA, 
    custom_data_collator, 
    custom_data_collator_perturbed,
    get_batch_loss
)
from data_generation.api import (
    GeminiEvaluator, 
    GPTEvaluator, 
    system_message, 
    user_message, 
    jobs
)

data_split = json.load(open("./dataset/split.json", "r"))

gpt_prompt = """You are an intelligent chatbot designed for evaluating the factual accuracy of generative outputs for question-answer pairs about fictitious entities.
Your task is to compare the predicted answer with the correct answer and determine if they are factually consistent. Here's how you can accomplish the task:
1. Focus on the meaningful match between the predicted answer and the correct answer.
2. Consider synonyms or paraphrases as valid matches.
3. Evaluate the correctness of the prediction compared to the answer.
4. Please do not consider the difference in sentence style between the correct answer and the predicted answer, but only judge whether the predicted answer makes sense based on factual accuracy.
5. If there is something in the predicted answer that is not in the correct answer, then it is considered to be hallucination.

The score should range from 0 to 1. A larger score means a better answer. The score should be a float number with 2 decimal places. For example, 0.51, 0.99, 0.00, 0.76, etc.
In additional to this, I would like you to be able to extract some key words from the question and the correct answer, which are considered to be the key to answering the question correctly, and a prediction tends to score higher if  the prediction is able to include these key words.
Please first output a single line containing only one value indicating the scores for the predicted answer.
In the subsequent line, please provide some key words of the question and correct answers.
In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

Question: {question}
Correct Answer: {answer}
Prediction: {prediction}

Outputs (include score, key words, explanation):"""

def eval_exact_match(pred, gt, keywords):
    score = 0.0
    for key in keywords:
        if key.lower() in pred.lower():
            score += 1.0 / len(keywords)
    return  min(1.0, score)

def eval_perturbation_ratio(cfg, tokenizer, eval_dataloader, perturb_dataloader, model):
    eval_logs = {}
    i = 0
    pbar = tqdm(total=len(eval_dataloader))
    for batch, perturb_batch in tqdm(zip(eval_dataloader, perturb_dataloader)):
        pbar.update(1)
        category = batch.pop("category")
        if len(perturb_batch['input_ids'].shape) > 2:
            bsz, seq_len = perturb_batch['input_ids'].shape[0:2]
        else:
            bsz = perturb_batch['input_ids'].shape[0]
            seq_len = 1
        
        llama_vision_inputs = {}
        if "cross_attention_mask" in perturb_batch.keys():
            llama_vision_inputs['cross_attention_mask'] = perturb_batch['cross_attention_mask'].view(bsz*seq_len, *perturb_batch['cross_attention_mask'].shape[2:])
            llama_vision_inputs['aspect_ratio_ids'] = perturb_batch['aspect_ratio_ids'].view(bsz*seq_len, *perturb_batch['aspect_ratio_ids'].shape[2:])
            llama_vision_inputs['aspect_ratio_mask'] = perturb_batch['aspect_ratio_mask'].view(bsz*seq_len, *perturb_batch['aspect_ratio_mask'].shape[2:])
            perturb_batch = {
                "input_ids": perturb_batch['input_ids'].view(bsz*seq_len, -1), 
                "labels": perturb_batch['labels'].view(bsz*seq_len, -1), 
                "attention_mask": perturb_batch['attention_mask'].view(bsz*seq_len, -1),
                "pixel_values": perturb_batch['pixel_values'].view(bsz*seq_len, 1, *perturb_batch['pixel_values'].shape[2:]),
            }

            perturb_batch.update(llama_vision_inputs)

        else:
            perturb_batch = {
                "input_ids": perturb_batch['input_ids'].view(bsz*seq_len, -1), 
                "labels": perturb_batch['labels'].view(bsz*seq_len, -1), 
                "attention_mask": perturb_batch['attention_mask'].view(bsz*seq_len, -1),
                "pixel_values": perturb_batch['pixel_values'].view(bsz*seq_len, *perturb_batch['pixel_values'].shape[2:]),
            }

        indices = [i * cfg.perturb_batch_size + j for j in range(cfg.perturb_batch_size)]
        
        #send to device
        for k, v in batch.items():
            batch[k] = v.to(model.device)
        for k, v in perturb_batch.items():
            perturb_batch[k] = v.to(model.device)
        
        with torch.no_grad():
            outputs = model(**batch)
            perturb_outputs = model(**perturb_batch)
        
        logits, perturb_logits = outputs.logits, perturb_outputs.logits
       
        labels = batch['labels']
        labels = labels[labels != -100].unsqueeze(0)
        logits = logits[:, -labels.shape[1]:, :]
        gt_loss = get_batch_loss(logits, labels)


        label_list, max_length = [], 0
        perturb_labels = perturb_batch['labels']
        for l in range(perturb_labels.shape[0]):
            label_tmp = perturb_labels[l]
            label_tmp = label_tmp[label_tmp != -100].unsqueeze(0)
            max_length = max(max_length, label_tmp.shape[1])
            label_list.append(label_tmp)

        perturb_loss = []
        for l in range(perturb_labels.shape[0]):
            label_tmp = label_list[l]
            current_length = label_tmp.shape[1]
            shifted = max_length - current_length
            if shifted == 0:
                logits_tmp = perturb_logits[l, -label_tmp.shape[1]:, :].unsqueeze(0)
            else:
                logits_tmp = perturb_logits[l, -label_tmp.shape[1]-shifted:-shifted, :].unsqueeze(0)
            perturb_loss.append(get_batch_loss(logits_tmp, label_tmp))
       
        perturb_loss = torch.tensor(perturb_loss).unsqueeze(0).to(model.device)
        num_token_gt = (batch['labels']!=-100).sum(-1)
        num_token_perturb = (perturb_batch['labels']!=-100).view(bsz, seq_len, -1).sum(-1)

        mean_perturb_loss = perturb_loss.mean(dim=1)
        ratio = (mean_perturb_loss - gt_loss).mean()

        # eval_logs["perplexity delta"] = eval_logs.get("perplexity delta", []) + [ratio.item()]

        # eval_logs['ground_truth_loss'] = eval_logs.get('ground_truth_loss', []) + [gt_loss.mean().item()]
        # eval_logs['perturb_loss'] = eval_logs.get('perturb_loss', []) + [mean_perturb_loss.mean().item()]

        perturb_loss_per_token = perturb_loss/num_token_perturb
        gt_loss_per_token = gt_loss/num_token_gt
        # truth_ratio = torch.exp(-1 * perturb_loss_per_token).mean(-1) / torch.exp(-1 * gt_loss_per_token)
        truth_ratio = torch.exp(gt_loss_per_token - perturb_loss_per_token.mean(-1))
 
    
        # zip index and each stat into a dict
        perturb_loss_per_token = dict(zip(indices, perturb_loss_per_token.cpu().numpy().tolist()))
        gt_loss_per_token = dict(zip(indices, gt_loss_per_token.cpu().numpy().tolist()))
        truth_ratio = dict(zip(indices, truth_ratio.cpu().numpy().tolist()))
        gt_loss = dict(zip(indices, gt_loss.cpu().numpy().tolist()))
        perturb_loss = dict(zip(indices, perturb_loss.cpu().numpy().tolist()))
        num_token_gt = dict(zip(indices, num_token_gt.cpu().numpy().tolist()))
        num_token_perturb = dict(zip(indices, num_token_perturb.cpu().numpy().tolist()))


        # merge dicts
        if 'average_perturb_loss' not in eval_logs:
            eval_logs['average_perturb_loss'] = {}
        if 'avg_paraphrased_loss' not in eval_logs:
            eval_logs['avg_paraphrased_loss'] = {}
        if 'truth_ratio' not in eval_logs:
            eval_logs['truth_ratio'] = {}
        if 'paraphrased_loss' not in eval_logs:
            eval_logs['paraphrased_loss'] = {}
        if 'perturb_loss' not in eval_logs:
            eval_logs['perturb_loss'] = {}
        if 'num_token_paraphrased' not in eval_logs:
            eval_logs['num_token_paraphrased'] = {}
        if 'num_token_perturb' not in eval_logs:
            eval_logs['num_token_perturb'] = {}

        eval_logs['average_perturb_loss'].update(perturb_loss_per_token)
        eval_logs['avg_paraphrased_loss'].update(gt_loss_per_token)
        eval_logs['truth_ratio'].update(truth_ratio)
        eval_logs['paraphrased_loss'].update(gt_loss)
        eval_logs['perturb_loss'].update(perturb_loss)
        eval_logs['num_token_paraphrased'].update(num_token_gt)
        eval_logs['num_token_perturb'].update(num_token_perturb)

        i += 1
    
    gc.collect()
    torch.cuda.empty_cache()

    return eval_logs

def get_dataloader(cfg, eval_task, tokenizer, image_processor, processor, data_path, split, question_key, answer_key, base_answer_key, perturbed_answer_key, paraphrased_question_key=None):
    
    torch_format_dataset = MMDatasetQA(  
        config=cfg,
        tokenizer=tokenizer, 
        image_processor=image_processor,
        data_path=data_path,
        max_length=cfg.generation.max_length, 
        split=split, 
        question_key=question_key, 
        answer_key=answer_key,
        processor=processor,
    ) 

        
    base_torch_format_dataset = MMDatasetQA(
        config=cfg,
        tokenizer=tokenizer, 
        image_processor=image_processor,
        data_path=data_path,
        max_length=cfg.generation.max_length, 
        split=split, 
        question_key=question_key, 
        answer_key=base_answer_key,
        processor=processor,
    )
    

    robust_torch_format_dataset = MMDatasetQA(  
        config=cfg,
        tokenizer=tokenizer, 
        image_processor=image_processor,
        data_path=data_path,
        max_length=cfg.generation.max_length, 
        split=split, 
        question_key=paraphrased_question_key, 
        answer_key=answer_key,
        processor=processor,
    ) 

  

    perturb_torch_format_dataset = MMDatasetQA(
        config=cfg,
        tokenizer=tokenizer, 
        image_processor=image_processor,
        data_path=data_path,
        max_length=cfg.generation.max_length, 
        split=split, 
        question_key=question_key, 
        answer_key=perturbed_answer_key,
        processor=processor,
    )

    eval_dataloader = DataLoader(
        torch_format_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
        shuffle=False,
        collate_fn=custom_data_collator(tokenizer=tokenizer),
    )

    base_eval_dataloader = DataLoader(
        base_torch_format_dataset,
        batch_size=cfg.perturb_batch_size,
        num_workers=cfg.workers,
        shuffle=False,
        collate_fn=custom_data_collator(tokenizer=tokenizer),
    )

    robust_eval_dataloader = DataLoader(
        robust_torch_format_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
        shuffle=False,
        collate_fn=custom_data_collator(tokenizer=tokenizer),
    )



    perturb_dataloader = DataLoader(
        perturb_torch_format_dataset,
        batch_size=cfg.perturb_batch_size,
        num_workers=cfg.workers,
        shuffle=False,
        collate_fn=custom_data_collator_perturbed(tokenizer=tokenizer),
    )

    


    return eval_dataloader, base_eval_dataloader, robust_eval_dataloader, perturb_dataloader


def get_all_evals(cfg, model, tokenizer, image_processor, eval_task, split, eval_dataloader, base_eval_dataloader, robust_eval_dataloader, perturb_dataloader, normalize_gt=False, model_cfg=None, metric_list=[]):
    eval_logs = {}
    gen_outputs = []
    ground_truths = []
    input_strings = []
    all_categories = []
    all_indices = []
    if "ape" in metric_list:
        pbar = tqdm(total=len(robust_eval_dataloader))
        for i, batch in enumerate(robust_eval_dataloader):
            pbar.update(1)
            category = batch.pop("category")
            all_categories.extend(category)
            for k, v in batch.items():
                batch[k] = v.to(model.device)

            with torch.no_grad():
                outputs = model(**batch)
                input_string, gen_output, gt = run_generation(cfg, batch, model, tokenizer=tokenizer)
                gen_outputs.extend(gen_output)
                ground_truths.extend(gt)
                input_strings.extend(input_string) 

            try:
                with open(cfg.data_path[0], "r") as f:
                    data = json.load(f)
            except:
                with open(cfg.data_path[0], "r") as f:
                    data = [json.loads(line) for line in f.readlines()]

            samples = []
            data = data[:400] ### choose 400 person for evaluation
            if split is not None:
                if split in data_split.keys():
                    data = [line for line in data if line['unique_id'] in data_split[split]]
                else:
                    print(
                        f"Incorrect dataset split name: {split}!"
                    )

            # data = data[:10] #TODO: to delete this line
            for line in data:
                qa_list = line['qa_list']
                for qa in qa_list:
                    qa.update(label="human_face")
                    qa.update(image_path=line['image_path'])
                    for _ in range(3): # three robust questions for evaluation
                        samples.append(qa)
            
            print(
                f"Keyword item number: {len(samples)}"
            )
            if 'exact_match' not in eval_logs:
                eval_logs['exact_match'] = []

            item = samples[i] ### note that don't shuffle the dataloader
            keywords = item['keywords']
            gt, gen = ground_truths[-1], gen_outputs[-1]
            meta_keywords = keywords
            # for item in keywords:
            #     meta_keywords.extend(item.lower().split(" "))

            eval_logs['exact_match'].append(eval_exact_match(gen.lower(), gt.lower(), meta_keywords))    
            
            print(
                f"exact_match: {eval_logs['exact_match'][-1]}"
            )

        return eval_logs
      
    eval_logs.update(eval_perturbation_ratio(cfg, tokenizer, base_eval_dataloader, perturb_dataloader, model))
    model_name = "gpt"
    if model_name == "gemini":
        agent = GeminiEvaluator(api_key="")
    elif model_name == "gpt":
        agent = GPTEvaluator(api_key="", model="gpt-4o-mini", max_tokens=20)
    pbar = tqdm(total=len(eval_dataloader))
    for i, batch in enumerate(eval_dataloader):
        pbar.update(1)
        category = batch.pop("category")
        all_categories.extend(category)
        for k, v in batch.items():
            batch[k] = v.to(model.device)

        indices = [cfg.batch_size * i + j for j in range(cfg.batch_size)]
        all_indices.extend(indices)

        with torch.no_grad():
            outputs = model(**batch)
            input_string, gen_output, gt = run_generation(cfg, batch, model, tokenizer=tokenizer)
            gen_outputs.extend(gen_output)
            ground_truths.extend(gt)
            input_strings.extend(input_string)      
            
        logits = outputs.logits
        labels = batch['labels']
        labels = labels[labels != -100].unsqueeze(0)
        logits = logits[:, -labels.shape[1]:, :]
        
        log_probs = F.log_softmax(logits[0, :], dim=-1)        
        top5_values, top5_indices = torch.topk(log_probs, k=5, dim=-1)
    
        gt_loss = get_batch_loss(logits,labels)
        num_token_gt = (batch['labels']!=-100).sum(-1)
        gt_loss_per_token = gt_loss/num_token_gt
        print(outputs.loss, gt_loss, gt_loss_per_token, num_token_gt)
      
        if 'avg_gt_loss' not in eval_logs:
            eval_logs['avg_gt_loss'] = {}
        if 'gt_loss' not in eval_logs:
            eval_logs['gt_loss'] = {}
        if 'num_token_gt' not in eval_logs:
            eval_logs['num_token_gt'] = {}
        if 'generated_text' not in eval_logs:
            eval_logs['generated_text'] = {}
        if 'mink' not in eval_logs:
            eval_logs['mink'] = []
            eval_logs['loss'] = []
            eval_logs['zlib'] = []
        if 'mink++' not in eval_logs:
            eval_logs['mink++'] = []
        if 'gpt' not in eval_logs:
            eval_logs['gpt'] = []
        if 'exact_match' not in eval_logs:
            eval_logs['exact_match'] = []
        
        # print(gt_loss.shape, num_token_gt.shape)

        eval_logs['avg_gt_loss'].update(dict(zip(indices, gt_loss_per_token.cpu().numpy().tolist())))
        eval_logs['gt_loss'].update(dict(zip(indices, gt_loss.cpu().numpy().tolist())))
        eval_logs['num_token_gt'].update(dict(zip(indices, num_token_gt.cpu().numpy().tolist())))
        eval_logs['generated_text'].update(dict(zip(indices, zip(input_string, gen_output, gt, category))))
        
        if "mink" in metric_list:
            loss, logits = outputs[:2]
            labels = batch['labels']
            labels = labels[labels != -100][1:].unsqueeze(0)
            logits = logits[:, -labels.shape[1]-1: -1, :]
            text = tokenizer.decode(labels[0])

            ll = -loss.item() # log-likelihood
            eval_logs['loss'].append(ll)
            eval_logs['zlib'].append(ll / len(zlib.compress(bytes(text, 'utf-8'))))

            # mink
            labels = labels[0].unsqueeze(-1)
            probs = F.softmax(logits[0, :], dim=-1)
            log_probs = F.log_softmax(logits[0, :], dim=-1)        
            # top5_values, top5_indices = torch.topk(log_probs, k=5, dim=-1)
            # print("Top 5 values for each token:\n", top5_values)
            # print("Top 5 indices for each token:\n", top5_indices)

            token_log_probs = log_probs.gather(dim=-1, index=labels[:,:]).squeeze(-1)
            mu = (probs * log_probs).sum(-1)
            sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
    
            ## mink
            mink_scores = []
            weights = [0.3, 0.3, 0.2, 0.1, 0.1]
            for ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
                k_length = int(len(token_log_probs) * ratio)
                topk = np.sort(token_log_probs.cpu())[:k_length]
                mink_scores.append(np.exp(np.mean(topk)).item())
            
            eval_logs[f'mink'].append(sum([score * w for score, w in zip(mink_scores, weights) if not math.isnan(score)]))
            
            mink_plus_plus_scores = []
            mink_plus = (token_log_probs - mu) / sigma.sqrt()
            weights = [0.3, 0.3, 0.2, 0.1, 0.1]
            for ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
                k_length = int(len(mink_plus) * ratio)
                topk = np.sort(mink_plus.cpu())[:k_length]
                mink_plus_plus_scores.append(np.exp(np.mean(topk)).item())
            
            eval_logs[f'mink++'].append(sum([score * w for score, w in zip(mink_plus_plus_scores, weights) if not math.isnan(score)]))
            
            print(
                f"mink++: {eval_logs['mink++'][-1]}"
            )

        if "exact_match" in metric_list:
            try:
                with open(cfg.data_path[0], "r") as f:
                    data = json.load(f)
            except:
                with open(cfg.data_path[0], "r") as f:
                    data = [json.loads(line) for line in f.readlines()]

            samples = []
            data = data[:400] ### choose 400 person for evaluation
            if split is not None:
                if split in data_split.keys():
                    data = [line for line in data if line['unique_id'] in data_split[split]]
                else:
                    print(
                        f"Incorrect dataset split name: {split}!"
                    )
            for line in data:
                qa_list = line['qa_list']
                for qa in qa_list:
                    qa.update(label="human_face")
                    qa.update(image_path=line['image_path'])
                    samples.append(qa)

            item = samples[i] ### note that don't shuffle the dataloader
            keywords = item['keywords']
            gt, gen = ground_truths[-1], gen_outputs[-1]
            meta_keywords = keywords
            # for item in keywords:
            #     meta_keywords.extend(item.lower().split(" "))


            eval_logs['exact_match'].append(eval_exact_match(gen.lower(), gt.lower(), meta_keywords))    
            print(
                f"exact_match: {eval_logs['exact_match'][-1]}"
            )
            if eval_logs['exact_match'][-1] != 0:
                print(eval_logs['exact_match'][-1], gen, meta_keywords)



        if "gpt" in metric_list:
            question = input_strings[0].replace(
                model_cfg['question_start_tag'].replace("\n", ""), "").replace(
                model_cfg['question_end_tag'].replace("\n", ""), "").replace(
                model_cfg['system_tag'].replace("\n", ""), "").replace(
                model_cfg['answer_tag'].replace("\n", ""), "").strip(" ").strip("\n").strip(" ")
            gt, gen = ground_truths[-1], gen_outputs[-1]

            if len(gen) <= 5:
                eval_logs['gpt'].append(0.0)
            else:
                question = {
                    "prompted_system_content": "",
                    "prompted_content": gpt_prompt.format(question=question, answer=gt, prediction=gen),
                    "image_list": None,
                }
                response = agent.generate_answer(question)
                
                try:
                    score = response['prediction'].split("\n")[0].strip(" ")
                    if ":" in score:
                        score = score[score.find(":"):].strip(":").strip(" ")
                    if "**" in score:
                        score = score.strip("**").strip(" ")
                    score = float(score)
                    eval_logs['gpt'].append(score)
                except:
                    eval_logs['gpt'].append(0.0)

            print(
                f"gpt score: {eval_logs['gpt'][-1]}"
            )


    gc.collect()
    torch.cuda.empty_cache()

    eval_logs.update(eval_rouge_recall(gen_outputs, ground_truths, all_indices))
  
    if normalize_gt:
        avg_gt_loss = eval_logs['avg_gt_loss']
        avg_perturb_loss = eval_logs['average_perturb_loss']
        data_indices = avg_gt_loss.keys()
        normalized_gt_loss = {}
        for idx in data_indices:
            truth_prob = np.exp(-1 * avg_gt_loss[idx])
            perturb_prob = np.exp(-1 * np.array(avg_perturb_loss[idx]))
            all_prob = np.array([truth_prob, *perturb_prob])
            normalized_gt_prob = truth_prob / all_prob.sum()
            normalized_gt_loss[idx] = -1 * np.log(normalized_gt_prob)

        eval_logs['normalized_gt_loss'] = normalized_gt_loss

    return eval_logs



@hydra.main(version_base=None, config_path="config", config_name="eval_everything")
def main(cfg):
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    tokenizer.pad_token = tokenizer.eos_token
    max_length = 500
    batch_size = cfg.batch_size

    model, processor = None, None
    if "llava" in cfg.model_path:
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        model = LlavaForConditionalGeneration.from_pretrained(cfg.model_path, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
        if cfg.LoRA.r != 0:
            target_modules=r'.*language_model.*\.(up_proj|k_proj|linear_2|down_proj|v_proj|q_proj|o_proj|gate_proj|linear_1)'
    elif "llama-3.2" in cfg.model_path.lower():
        model = MllamaForConditionalGeneration.from_pretrained(cfg.model_path, torch_dtype=torch.bfloat16)
        processor = AutoProcessor.from_pretrained(cfg.model_path)
        image_processor = processor.image_processor
        tokenizer = processor.tokenizer
        if cfg.LoRA.r != 0:
            target_modules=r'.*language_model.*\.(up_proj|k_proj|down_proj|v_proj|q_proj|o_proj|gate_proj)'


    if cfg.LoRA.r != 0:
        config = LoraConfig(
            r=cfg.LoRA.r, 
            lora_alpha=cfg.LoRA.alpha, 
            target_modules=target_modules, 
            lora_dropout=cfg.LoRA.dropout,
            bias="none", 
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)

        if cfg.LoRA.lora_path is not None:
            model.load_state_dict(torch.load(cfg.LoRA.lora_path), strict=False)
            model.merge_and_unload() 
            path = cfg.LoRA.lora_path.replace("/checkpoint.pt", "")
            cfg.save_dir = os.path.join(path, "eval_results")

            print(
                f"Successful loading LoRA weights from {cfg.LoRA.lora_path}!"
            )

        elif cfg.ckpt_path is not None:    
            model.load_state_dict(torch.load(cfg.ckpt_path), strict=False)
            path = cfg.ckpt_path.replace("/checkpoint.pt", "")
            cfg.save_dir = os.path.join(path, "eval_results")

            print(
                f"Successful loading weights from {cfg.ckpt_path}!"
            )
    
    model.half().cuda()

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
         
    aggregated_eval_logs = {}
    for i, (folder, split, question_key, robust_question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key, metric_list) in enumerate(zip(cfg.data_path, cfg.split_list, cfg.question_key, cfg.robust_question_key, cfg.answer_key, cfg.eval_task, cfg.base_answer_key, cfg.perturbed_answer_key, cfg.robust_eval)):
        print(f'Working on eval task {eval_task} with split {split}')
        save_filename = os.path.join(cfg.save_dir, f"{split}_{eval_task}.json")
        print(f"Save logs into {save_filename}!")
        if os.path.exists(save_filename) and not cfg.overwrite:
            print(f"Skipping {eval_task} because {save_filename} already exists")
            with open(save_filename, "r") as f:
                eval_logs = json.load(f)
        else:
            eval_logs = {}
            eval_dataloader, base_eval_dataloader, robust_eval_dataloader, perturb_dataloader = get_dataloader(cfg, eval_task, tokenizer, image_processor, processor, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key, robust_question_key)
            normalize_gt = False 
            if 'eval_retain_log' not in eval_task:
                normalize_gt = True

            eval_logs = get_all_evals(cfg, model, tokenizer, image_processor, eval_task, split, eval_dataloader, base_eval_dataloader, robust_eval_dataloader, perturb_dataloader, normalize_gt=normalize_gt, model_cfg=model_cfg, metric_list=metric_list)
           
            with open(save_filename, "w") as f:
                # pretty write json to f
                json.dump(eval_logs, f, indent=4)

        aggregated_eval_logs[f'{eval_task}.json'] = eval_logs

    aggregated_eval_log_filename = os.path.join(cfg.save_dir, f"{split}_eval_log_aggregated.json")
            
    with open(aggregated_eval_log_filename, "w") as f:
        # pretty write json to f
        json.dump(aggregated_eval_logs, f, indent=4)


def eval_accuracy(logits, labels):
    preds =logits.argmax(-1)
    shifted_labels = labels[..., 1:].contiguous()
    # the places where labels is -100 should be ignored in the accuracy computation
    mask = (shifted_labels != -100)
    acc = (preds[..., :-1] == shifted_labels).float()
    acc *= mask.float()
    acc = acc.sum() / mask.float().sum()

    return {"eval accuracy": acc.item()}


def run_generation(cfg, batch, model, tokenizer):
    input_ids = batch["input_ids"]
    pixel_values = batch['pixel_values']
    aspect_ratio_ids, aspect_ratio_mask, cross_attention_mask = None, None, None
    if "aspect_ratio_ids" in batch.keys():
        aspect_ratio_ids = batch['aspect_ratio_ids']
        aspect_ratio_mask = batch['aspect_ratio_mask']
        cross_attention_mask = batch['cross_attention_mask']

    input_strings = tokenizer.batch_decode(input_ids)

    model_config = get_model_identifiers_from_yaml(cfg.model_family)
    question_start_tag = model_config['question_start_tag']
    answer_tag = model_config['answer_tag']
    answer_tag = answer_tag.replace("\n", "") 

    ground_truth = [s.split(answer_tag)[1].strip(" ") for s in input_strings]
    input_strings = [s.split(answer_tag)[0].strip(" ") for s in input_strings]
    input_strings = [s + answer_tag for s in input_strings]
    
    if "llava_phi" in cfg.model_family:
        input_strings = [s.replace(question_start_tag, f"{question_start_tag} <image>") for s in input_strings]
        input_strings = [s.replace("<|user|>", "<|user|>\n") for s in input_strings]
        input_strings = [s.replace("<|end|>", "<|end|>\n") for s in input_strings]
    
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = 'left'
    left_pad_tokenizer.padding_size = 'longest'
    left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
    left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id

    inputs = left_pad_tokenizer.batch_encode_plus(input_strings, add_special_tokens=True, return_tensors='pt', padding=True).to(model.device)
    #now generate
    if aspect_ratio_ids is not None:
        out = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, pixel_values=pixel_values, aspect_ratio_ids=aspect_ratio_ids, aspect_ratio_mask=aspect_ratio_mask, cross_attention_mask=cross_attention_mask, max_new_tokens=cfg.generation.max_new_tokens, do_sample=False, use_cache=True, pad_token_id=left_pad_tokenizer.eos_token_id)
    else:
        out = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, pixel_values=pixel_values, max_new_tokens=cfg.generation.max_new_tokens, do_sample=False, use_cache=True, pad_token_id=left_pad_tokenizer.eos_token_id)
    strs = left_pad_tokenizer.batch_decode(out[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    strs = [s[:s.find(".")+1] for s in strs]
    return input_strings, strs, ground_truth

def eval_bleu(gen_outputs, ground_truths):

    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    rouge_res = rouge.compute(predictions=gen_outputs, references=ground_truths)
    bleu_res = bleu.compute(predictions=gen_outputs, references=ground_truths)


    eval_result = {
        'rouge': rouge_res,
        'bleu': bleu_res,
    }
    return eval_result



def eval_rouge_recall(gen_outputs, ground_truths, indices):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_recall = {}
    rougeL_recall = {}
    for gen, gt, idx in zip(gen_outputs, ground_truths, indices):
        gen = gen[:gen.find(".")]
        rouge_scores = scorer.score(gt, gen)
        rouge1_recall[idx] = rouge_scores['rouge1'].recall
        rougeL_recall[idx] = rouge_scores['rougeL'].recall


    return {'rouge1_recall': rouge1_recall, 'rougeL_recall': rougeL_recall}

if __name__ == "__main__":
    main()