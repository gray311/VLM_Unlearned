import os
import json
import torch
import random
import math
import gc
from PIL import Image
from tqdm import tqdm
from rouge_score import rouge_scorer
from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    set_seed, 
    LlavaForConditionalGeneration, 
    AutoProcessor,
    CLIPImageProcessor
)

random.seed(233)

# model_path = "./models/final_ft_LORA_6_epochs_inst_lr0.0001_llava-v1.6-vicuna_full"

# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = LlavaForConditionalGeneration.from_pretrained(model_path, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
# model.to("cuda:0")

# for n, p in model.named_parameters():
#     print(n, p.shape)
# image_path = "./dataset/images/fictitious/cactus boxer/macavity6328_A_boxer_has_a_cactus_head_0492216d-7d42-442d-b46e-966d550893d4_1.png"
# image = Image.open(image_path)
# prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nWhat is the name of the object in the image? ASSISTANT:"
# image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
# image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(model.device)

# print(image_tensor.shape)
# print(prompt)
# text_input = tokenizer(prompt, return_tensors='pt')
# text_input = {k: v.to(model.device) for k, v in text_input.items()}
# print(text_input['input_ids'])

# inputs = {**text_input, "pixel_values": image_tensor}
# output = model.generate(**inputs, max_new_tokens=1024)

# print(tokenizer.decode(output[0], skip_special_tokens=True))

def main(eval_file):
    if eval_file is None:
        file = "./dataset/full.json"
        # model_path = "./models/final_ft_LORA_6_epochs_inst_lr0.0001_llava-v1.6-vicuna_full"
        model_path = "llava-hf/llava-v1.6-vicuna-7b-hf"
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = LlavaForConditionalGeneration.from_pretrained(model_path, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        model.half().to("cuda:0")
        
        with open(file, "r") as f:
            data = json.load(f)
            
        random.shuffle(data)
        print(
            f"Full dataset length (only include fictitious examples): {len(data)}."
        )
        eval_data = data[:]
        print(
            f"Subset length of the full dataset for evaluation: {len(eval_data)}."
        )
        
        nlls = []
        with open("./outputs/finetune_results.json", "w") as f:
            for line in tqdm(eval_data):
                image_path = line['image_path']
                image = Image.open(image_path)
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(model.device)
                qa_list = line['question_and_answer']
                
                for qa in qa_list:
                    question, answer = qa['q'], qa['a']
                    prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{question} ASSISTANT:"
                    text_input = tokenizer(prompt, return_tensors='pt')
                    text_input = {k: v.to(model.device) for k, v in text_input.items()}
                    inputs = {**text_input, "pixel_values": image_tensor}
                    
                    output = model.generate(**inputs, max_new_tokens=25)
                    
                    prediction = tokenizer.decode(output[0], skip_special_tokens=True)
                    prediction = prediction[prediction.rfind("ASSISTANT:") + len("ASSISTANT:"):].strip(" ")
                
                    outputs = {
                            "question": question,
                            "answer": answer,
                            "prediction": prediction 
                        }
                    
                    print(outputs)
                    f.write(f"{json.dumps(outputs)}\n")
                    
                    ### compute PPL ###
                    conversation = prompt + " " + answer
                    text_input = tokenizer(conversation, return_tensors='pt')
                    labels = text_input['input_ids'].clone()
                    target = labels[0]
                    instruction = tokenizer(prompt, return_tensors='pt')
                    target[: len(instruction["input_ids"][0])] = -100
                    labels = target.unsqueeze(0)
                    
                    target[target==-100] = 0
                    text_input.update(labels=labels)
                    
                    text_input = {k: v.to(model.device) for k, v in text_input.items()}
                    inputs = {**text_input, "pixel_values": image_tensor}
                    with torch.no_grad():
                        neg_log_likelihood = model(**inputs).loss
                    print(neg_log_likelihood)
                    nlls.append(neg_log_likelihood)
                    
              
        print(
            f"Avg PPL scores: {torch.exp(torch.stack(nlls).mean()) }"
        )           
    else:
        with open(eval_file, "r") as f:
            data = json.load(f)
            
        rougeL_list = []
        for line in data:
            pred, gt = line['prediction'], line['answer']
            pred = pred[:pred.find(".")]
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(gt, pred)
            rougeL_list.append(rouge_scores['rougeL'].precision)
        
        print(
            f"Avg RougeL scores: {sum(rougeL_list) / len(rougeL_list)}"
        )
            
        
    
if __name__ == "__main__":
    eval_file = None
    main(eval_file)
        