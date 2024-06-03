import os 
import sys
import hydra
import json
import gc
from tqdm import tqdm
import torch.nn as nn
import numpy as np 
import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    set_seed, 
    LlavaForConditionalGeneration, 
    AutoProcessor,
    CLIPImageProcessor
)
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
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



def eval_perturbation_ratio(cfg, tokenizer, eval_dataloader, perturb_dataloader, model):
    eval_logs = {}
    i = 0
    image_token_id = 32000 if "llava" in cfg.model_family else 32000
    pbar = tqdm(total=len(eval_dataloader))
    for batch, perturb_batch in tqdm(zip(eval_dataloader, perturb_dataloader)):
        pbar.update(1)
        category = batch.pop("category")
        if len(perturb_batch['input_ids'].shape) > 2:
            bsz, seq_len = perturb_batch['input_ids'].shape[0:2]
        else:
            bsz = perturb_batch['input_ids'].shape[0]
            seq_len = 1

        perturb_batch = {
            "input_ids": perturb_batch['input_ids'].view(bsz*seq_len, -1), 
            "labels": perturb_batch['labels'].view(bsz*seq_len, -1), 
            "attention_mask": perturb_batch['attention_mask'].view(bsz*seq_len, -1),
            "pixel_values": perturb_batch['pixel_values'].view(bsz*seq_len, 3, 336, 336),
        }


        indices = [i * cfg.perturb_batch_size + j for j in range(cfg.perturb_batch_size)]

        _, image_token_start= (batch['input_ids'] == image_token_id).nonzero(as_tuple=True)
        _, image_token_start_perturb = (perturb_batch['input_ids'] == image_token_id).nonzero(as_tuple=True)
    
        #send to device
        for k, v in batch.items():
            batch[k] = v.to(model.device)
        for k, v in perturb_batch.items():
            perturb_batch[k] = v.to(model.device)

        with torch.no_grad():
            outputs = model(**batch)
            perturb_outputs = model(**perturb_batch)
        
        logits, perturb_logits = outputs.logits, perturb_outputs.logits
        logits = torch.cat((logits[:, :image_token_start[0] + 1, :], logits[:, image_token_start[0] + 576:, :]), dim=1)
        perturb_logits = torch.cat((perturb_logits[:, :image_token_start_perturb[0] + 1, :], perturb_logits[:, image_token_start_perturb[0] + 576:, :]), dim=1)
        
        gt_loss = get_batch_loss(logits, batch['labels'])
        perturb_loss = get_batch_loss(perturb_logits, perturb_batch['labels']).view(bsz, seq_len)
        
        
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

def get_dataloader(cfg, eval_task, tokenizer, image_processor, folder, file, question_key, answer_key, base_answer_key, perturbed_answer_key):
    
    data_path = os.path.join(folder, cfg.split, f"{file}.json")
    torch_format_dataset = MMDatasetQA(  
        config=cfg,
        tokenizer=tokenizer, 
        image_processor=image_processor,
        data_path=data_path,
        max_length=cfg.generation.max_length, 
        split=file, 
        question_key=question_key, 
        answer_key=answer_key
    ) 
    
    base_torch_format_dataset = MMDatasetQA(
        config=cfg,
        tokenizer=tokenizer, 
        image_processor=image_processor,
        data_path=data_path,
        max_length=cfg.generation.max_length, 
        split=file, 
        question_key=question_key, 
        answer_key=base_answer_key
    )

    perturb_torch_format_dataset = MMDatasetQA(
        config=cfg,
        tokenizer=tokenizer, 
        image_processor=image_processor,
        data_path=data_path,
        max_length=cfg.generation.max_length, 
        split=file, 
        question_key=question_key, 
        answer_key=perturbed_answer_key
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

    perturb_dataloader = DataLoader(
        perturb_torch_format_dataset,
        batch_size=cfg.perturb_batch_size,
        num_workers=cfg.workers,
        shuffle=False,
        collate_fn=custom_data_collator_perturbed(tokenizer=tokenizer),
    )

    return eval_dataloader, base_eval_dataloader, perturb_dataloader

def get_all_evals(cfg, model, tokenizer, image_processor, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader, normalize_gt=False):
    eval_logs = {}

    gen_outputs = []
    ground_truths = []
    input_strings = []
    all_categories = []
    all_indices = []
    image_token_id = 32000 if "llava" in cfg.model_family else 32000
    
    eval_logs.update(eval_perturbation_ratio(cfg, tokenizer, base_eval_dataloader, perturb_dataloader, model))

    pbar = tqdm(total=len(eval_dataloader))
    for i, batch in enumerate(eval_dataloader):
        pbar.update(1)
        category = batch.pop("category")
        all_categories.extend(category)
        for k, v in batch.items():
            batch[k] = v.to(model.device)

        indices = [cfg.batch_size * i + j for j in range(cfg.batch_size)]
        all_indices.extend(indices)

        _, image_token_start= (batch['input_ids'] == image_token_id).nonzero(as_tuple=True)
  
        with torch.no_grad():
            outputs = model(**batch)
            input_string, gen_output, gt = run_generation(cfg, batch, model, tokenizer=tokenizer)
            gen_outputs.extend(gen_output)
            ground_truths.extend(gt)
            input_strings.extend(input_string)      
            
        logits = outputs.logits
        logits = torch.cat((logits[:, :image_token_start[0] + 1, :], logits[:, image_token_start[0] + 576:, :]), dim=1)

        gt_loss = get_batch_loss(logits, batch['labels'])
 
        num_token_gt = (batch['labels']!=-100).sum(-1)
        gt_loss_per_token = gt_loss/num_token_gt

        print(outputs.loss, gt_loss, gt_loss_per_token)

        if 'avg_gt_loss' not in eval_logs:
            eval_logs['avg_gt_loss'] = {}
        if 'gt_loss' not in eval_logs:
            eval_logs['gt_loss'] = {}
        if 'num_token_gt' not in eval_logs:
            eval_logs['num_token_gt'] = {}
        if 'generated_text' not in eval_logs:
            eval_logs['generated_text'] = {}
        # print(gt_loss.shape, num_token_gt.shape)

        eval_logs['avg_gt_loss'].update(dict(zip(indices, gt_loss_per_token.cpu().numpy().tolist())))
        eval_logs['gt_loss'].update(dict(zip(indices, gt_loss.cpu().numpy().tolist())))
        eval_logs['num_token_gt'].update(dict(zip(indices, num_token_gt.cpu().numpy().tolist())))
        eval_logs['generated_text'].update(dict(zip(indices, zip(input_string, gen_output, gt, category))))
    
    gc.collect()
    torch.cuda.empty_cache()

    eval_logs.update(eval_rouge_recall(gen_outputs, ground_truths, all_indices))
    # eval_logs.update(eval_exact_match(gen_outputs, ground_truths, all_categories, all_indices)) 

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

    model = None
    if "llava" in cfg.model_path:
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        model = LlavaForConditionalGeneration.from_pretrained(cfg.model_path, attn_implementation="flash_attention_2", torch_dtype=torch.float16)

    if cfg.LoRA.r != 0:
        target_modules=r'.*language_model.*\.(up_proj|k_proj|linear_2|down_proj|v_proj|q_proj|o_proj|gate_proj|linear_1)'
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
    
    model.half().cuda()

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
         
    aggregated_eval_logs = {}
    for i, (folder, file, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key) in enumerate(zip(cfg.data_path, cfg.file_list, cfg.question_key, cfg.answer_key, cfg.eval_task, cfg.base_answer_key, cfg.perturbed_answer_key)):
        print(f'Working on eval task {eval_task} with split {cfg.split}')
        save_filename = os.path.join(cfg.save_dir, f"{cfg.split}_{eval_task}.json")
        print(f"Save logs into {save_filename}!")
        if os.path.exists(save_filename) and not cfg.overwrite:
            print(f"Skipping {eval_task} because {save_filename} already exists")
            with open(save_filename, "r") as f:
                eval_logs = json.load(f)
        else:
            eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(cfg, eval_task, tokenizer, image_processor, folder, file, question_key, answer_key, base_answer_key, perturbed_answer_key)
            normalize_gt = False 
            if 'eval_retain_log' not in eval_task:
                normalize_gt = True
            eval_logs = get_all_evals(cfg, model, tokenizer, image_processor, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader, normalize_gt=normalize_gt)

            with open(save_filename, "w") as f:
                # pretty write json to f
                json.dump(eval_logs, f, indent=4)

        aggregated_eval_logs[f'{eval_task}.json'] = eval_logs

    aggregated_eval_log_filename = os.path.join(cfg.save_dir, f"{cfg.split}_eval_log_aggregated.json")

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
    input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    model_config = get_model_identifiers_from_yaml(cfg.model_family)
    question_start_tag = model_config['question_start_tag']
    answer_tag = model_config['answer_tag']
    ground_truth = [s.split(answer_tag)[1].strip(" ") for s in input_strings]
    input_strings = [s.split(answer_tag)[0].strip(" ") for s in input_strings]
    input_strings = [s + answer_tag for s in input_strings]
    input_strings = [s.replace(question_start_tag, f"{question_start_tag} <image>") for s in input_strings]
        
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = 'left'
    left_pad_tokenizer.padding_size = 'longest'
    left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
    left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id

    inputs = left_pad_tokenizer.batch_encode_plus(input_strings, add_special_tokens=True, return_tensors='pt', padding=True).to(model.device)
    #now generate
    out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, pixel_values=pixel_values, max_new_tokens=cfg.generation.max_new_tokens, do_sample=False, use_cache=True, pad_token_id=left_pad_tokenizer.eos_token_id)
    strs = left_pad_tokenizer.batch_decode(out[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    # strs = [s[:s.find(".")] for s in strs]
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

def eval_exact_match(gen_outputs, ground_truths, categories, indices):
    em_scores = 0

    import re
    pattern = r'"([^"]+)"'

    for gen, gt, label, idx in zip(gen_outputs, ground_truths, categories, indices):
        match = re.search(pattern, gt)
        if match:
            name = match.group(1)
        else:
            name = None

        if isinstance(name, list):
            name = name[0]
            
        if " " in label:
            cate_1, cate_2 = label.split(" ")[0], label.split(" ")[-1]
        else:
            cate_1, cate_2 = label, label

        score = 0
        if name is not None and name.lower() in gen.lower():
            score += 1.0
        if label.lower() in gen.lower():
            score += 1.0  
        elif cate_1.lower() in gen.lower() and cate_2.lower() in gen.lower():
            score += 1.0
        elif cate_1.lower() in gen.lower() or cate_2.lower() in gen.lower():
            score += 0.5
        em_scores += min(1.0, score)
    
    return {"exact_match": em_scores / len(gen_outputs)}
       

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
