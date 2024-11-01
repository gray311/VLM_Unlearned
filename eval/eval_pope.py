import os
import json
import torch
import argparse
import pandas as pd
import pyarrow.parquet as pq
import base64
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    set_seed, 
    LlavaForConditionalGeneration, 
    AutoProcessor,
    CLIPImageProcessor
)
from peft import LoraConfig, get_peft_model
 
def base64_pil(base64_str):
    image = BytesIO(base64_str)
    image = Image.open(image)
    return image

def parse_pred_ans(pred_ans):
    pred_label = None
    if pred_ans in ["yes", "no"]:
        pred_label = pred_ans
    else:
        prefix_pred_ans = pred_ans[:4]

        if "yes" in prefix_pred_ans:
            pred_label = "yes"
        elif "no" in prefix_pred_ans:
            pred_label = "no"
        else:
            pred_label = "other"

    return pred_label 

def load_model(args):
    if "llava" in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = LlavaForConditionalGeneration.from_pretrained(args.model_path, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
        image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)
        
        if args.ckpt_path is not None and args.use_lora:
            target_modules=r'.*language_model.*\.(up_proj|k_proj|linear_2|down_proj|v_proj|q_proj|o_proj|gate_proj|linear_1)'
            config = LoraConfig(
                r=128, 
                lora_alpha=256, 
                target_modules=target_modules, 
                lora_dropout=0.05,
                bias="none", 
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, config)
            checkpoint_path = args.ckpt_path
            model_state = torch.load(checkpoint_path)
            model.load_state_dict(torch.load(checkpoint_path), strict=False)
            model.merge_and_unload() 

        elif args.ckpt_path:
            print(
                f"load weigths from {args.ckpt_path}!"
            )
            checkpoint_path = args.ckpt_path
            model_state = torch.load(checkpoint_path)
            model.load_state_dict(torch.load(checkpoint_path), strict=False)

        model.half().cuda()

    elif "instructblip" in args.model_name:
        model, tokenizer, image_processor = None, None, None 
    
    return model, tokenizer, image_processor
    
def get_text_inputs(model_name, tokenizer, question, image_tensor):
    if "llava_phi" in model_name:
        prompt = f"<|user|>\n<image>\n{question}<|end|>\n<|assistant|>\n"
        text_input = tokenizer(prompt, return_tensors='pt')
        inputs = {**text_input, "pixel_values": image_tensor}
    elif "llava" in model_name:
        prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{question} ASSISTANT:"
        text_input = tokenizer(prompt, return_tensors='pt')
        inputs = {**text_input, "pixel_values": image_tensor}
        
    elif "instructblip" in model_name:
        inputs = None 
        
    return inputs
        
        
def pope_forward(model_name, image, question, answer, model, tokenizer, image_processor):
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(model.device)
    outputs = []
    inputs = get_text_inputs(model_name, tokenizer, question, image_tensor)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    output = model.generate(**inputs, max_new_tokens=5)
    if "llava_phi" in model_name:
        prediction = tokenizer.decode(output[0])
        prediction = prediction[prediction.find("<|assistant|>") + len("<|assistant|>"): ].strip(" ")
    elif "llava" in model_name:
        prediction = tokenizer.decode(output[0], skip_special_tokens=True)
        prediction = prediction[prediction.rfind("ASSISTANT:") + len("ASSISTANT:"):].strip(" ")
    elif "instructblip" in model_name:
        prediction = None  
    outputs.append("\t".join([question.strip("\n"), answer.strip("\n"), prediction.strip("\n")]))
    print(outputs[-1])
    return question.strip("\n"), answer.strip("\n"), prediction.strip("\n")
    
def main(args):

    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.ckpt_path is not None:
        ckpt_name = args.ckpt_path.split("/")[-2].strip(" ")
        args.output_dir = os.path.join(args.output_dir, ckpt_name)
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
    
    is_eval = {"random": False, "popular": False, "adversarial": False}

        
    model, tokenizer, image_processor = load_model(args)
    
    from collections import defaultdict
    scores = defaultdict(float)
    for category in os.listdir(args.pope_dir):
        acc = 0
        if ".parquet" not in category: continue
        if args.model_name in category: continue
        if "llava" in category: continue
    
        task = category.split("-")[0].strip(" ")
        task = task.split("_")[-1].strip(" ")
        if is_eval[task]: continue 
        
        path = os.path.join(args.pope_dir, category)
        outputs = []
        table = pq.read_table(path)
        df = table.to_pandas()
        for i in tqdm(range(df.shape[0])):
            image_str = df.loc[i, "image"]['bytes']
            question = df.loc[i, "question"]
            answer = df.loc[i, "answer"]
            image = base64_pil(image_str)
            question, gt_ans, pred_ans = pope_forward(args.model_name, image, question, answer, model, tokenizer, image_processor)
            outputs.extend("\t".join([question, gt_ans, pred_ans]))
            
            gt_ans = gt_ans.lower()
            pred_ans = pred_ans.lower()
            pred_ans = parse_pred_ans(pred_ans)
            if pred_ans == gt_ans:
                acc += 1
        
        print(
            f"Accuracy on {category} of POPE: {acc} ({len(outputs)}), {acc / len(outputs)}."
        )
        scores[category] = acc / 3000
        
        with open(os.path.join(args.output_dir, f"{args.model_name}_{category}.txt"), "w") as f:
            for line in outputs:
                f.write(f"{line}\n") 
    print(scores)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pope_dir", 
        type=str, 
        default=None, 
        help=""
        )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="llava", 
        help=""
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="llava-hf/llava-v1.6-vicuna-7b-hf", 
        help=""
    )
    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        default=None, 
        help=""
    )
    parser.add_argument(
        "--vision_tower", 
        type=str, 
        default="openai/clip-vit-large-patch14-336", 
        help=""
    )
    parser.add_argument(
        "--use_lora", 
        type=bool, 
        default=False, 
        help=""
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help=""
    )
    args = parser.parse_args()
    main(args)
    
    
"""
python ./eval/eval_pope.py \
    --model_name llava_pt \
    --pope_dir ./eval/pope \
    --output_dir ./eval/pope
    
CUDA_VISIBLE_DEVICES=3  python ./eval/eval_pope.py \
    --model_name llava_ft \
    --model_path models/vlm_unlearned_ft_llava_v1.6_vicuna_7b \
    --ckpt_path models/vlm_unlearned-llava_vicuna_ckpt_lora_icd/icd_4e-05_exp4_10/step_24/checkpoint.pt \
    --pope_dir ./eval/pope \
    --output_dir ./eval/pope/llava_ft 


CUDA_VISIBLE_DEVICES=3  python ./eval/eval_pope.py \
    --model_name llava_phi_ft \
    --model_path models/vlm_unlearned_ft_llava_phi_3_mini \
    --ckpt_path models/vlm_unlearned-llava_phi_ckpt_lora_icd/icd_4e-05_exp4_10/checkpoint.pt \
    --pope_dir ./eval/pope \
    --output_dir ./eval/pope/llava_phi_ft 
"""



