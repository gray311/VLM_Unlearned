import os
import json
import torch
import argparse
from PIL import Image
from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    set_seed, 
    LlavaForConditionalGeneration, 
    AutoProcessor,
    CLIPImageProcessor
)
from peft import LoraConfig, get_peft_model

def load_model(args):
    if "llava" in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = LlavaForConditionalGeneration.from_pretrained(args.model_path, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
        image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)
        
        if args.ckpt_path is not None and args.use_lora:
            print(
                f"add lora from {args.ckpt_path}!"
            )
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
        
        
def mme_forward(model_name, img_path, img_name, text_path, model, tokenizer, image_processor):
    with open(text_path, "r") as f:
        data = [line for line in f.readlines()]
        
    image = Image.open(img_path)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(model.device)
    outputs = []
    for line in data:
        question, answer = line.split("\t")[0], line.split("\t")[-1]
        inputs = get_text_inputs(model_name, tokenizer, question, image_tensor)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        output = model.generate(**inputs, max_new_tokens=5)
        if "llava_phi" in model_name:
            prediction = tokenizer.decode(output[0])
            prediction = prediction[prediction.find("<|assistant|>") + len("<|assistant|>"): ].strip(" ")
            if "yes" in prediction.lower():
                prediction = "yes"
            else:
                prediction = "no"
        elif "llava" in model_name:
            prediction = tokenizer.decode(output[0], skip_special_tokens=True)
            prediction = prediction[prediction.rfind("ASSISTANT:") + len("ASSISTANT:"):].strip(" ")
        elif "instructblip" in model_name:
            prediction = None  
        outputs.append("\t".join([img_name, question.strip("\n"), answer.strip("\n"), prediction.strip("\n")]))
        print(outputs[-1])
    return outputs
    
def main(args):
    
    model, tokenizer, image_processor = load_model(args)
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.ckpt_path is not None:
        ckpt_name = args.ckpt_path.split("/")[-2].strip(" ")
        args.output_dir = os.path.join(args.output_dir, ckpt_name)
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
    
    for category in os.listdir(args.mme_dir):
        if ".txt" in category: continue
        path = os.path.join(args.mme_dir, category)
        outputs = []
        # if f"{category}.txt" in os.listdir(args.output_dir): continue
        # print(category)
        if "images" in os.listdir(path):
            for img_name in os.listdir(os.path.join(path, "images")):
                if ".png" not in img_name and ".jpg" not in img_name: continue
                img_path = os.path.join(path, "images", img_name)
                text_path = os.path.join(path, "questions_answers_YN", f"{img_name.split('.')[0]}.txt")
                output = mme_forward(args.model_name, img_path, img_name, text_path, model, tokenizer, image_processor)
                outputs.extend(output)
        else:
            for img_name in os.listdir(path):
                if ".png" not in img_name and ".jpg" not in img_name: continue
                img_path = os.path.join(path, img_name)
                text_path = os.path.join(path, f"{img_name.split('.')[0]}.txt")
                output = mme_forward(args.model_name, img_path, img_name, text_path, model, tokenizer, image_processor)
                outputs.extend(output)
                
        with open(os.path.join(args.output_dir, f"{category}.txt"), "w") as f:
            for line in outputs:
                f.write(f"{line}\n") 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mme_dir", 
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
python ./eval/eval_mme.py --mme_dir ./eval/MME_Benchmark_release_version/ --output_dir ./eval/eval_tool/llava_pt

CUDA_VISIBLE_DEVICES=0 python ./eval/eval_mme.py \
    --model_name llava \
    --model_path models/vlm_unlearned_ft_llava_v1.6_vicuna_7b \
    --ckpt_path models/vlm_unlearned-exp2_llava_phi_ckpt_lora/KL_4e-05_exp2_8/step_24/checkpoint.pt \
    --mme_dir ./eval/MME_Benchmark_release_version/ \
    --output_dir ./eval/eval_tool/llava_ft


CUDA_VISIBLE_DEVICES=6 python ./eval/eval_mme.py \
    --model_name llava_phi \
    --model_path models/vlm_unlearned_ft_llava_phi_3_mini \
    --ckpt_path models/vlm_unlearned-ablation_llava_phi_ckpt_idk/idk_0.0001_exp2_10_vision/checkpoint.pt \
    --mme_dir ./eval/MME_Benchmark_release_version/ \
    --output_dir ./eval/eval_tool/llava_phi_ft

python ./eval/eval_tool/calculation.py \
    --results_dir eval/eval_tool/llava_phi_ft/idk_0.0001_exp2_10_mm

0 2 3 5 6
"""



