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
    CLIPImageProcessor,
    MllamaForConditionalGeneration
)
from peft import LoraConfig, get_peft_model
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_model(args):
    if "llava" in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = LlavaForConditionalGeneration.from_pretrained(args.model_path, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
        image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)
        processor = None
        if args.ckpt_path is not None and args.use_lora:
            target_modules=r'.*language_model.*\.(up_proj|k_proj|linear_2|down_proj|v_proj|q_proj|o_proj|gate_proj|linear_1)'
    
    elif "llama-3.2" in args.model_name:
        model = MllamaForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
        processor = AutoProcessor.from_pretrained(args.model_path)
        image_processor = processor.image_processor
        tokenizer = processor.tokenizer
        if args.ckpt_path is not None and args.use_lora:
            target_modules=r'.*language_model.*\.(up_proj|k_proj|down_proj|v_proj|q_proj|o_proj|gate_proj)'
    
    elif "instructblip" in args.model_name:
        model, tokenizer, image_processor = None, None, None 

    if args.ckpt_path is not None and args.use_lora:
        print(
            f"add lora from {args.ckpt_path}!"
        )
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
    
    return model, tokenizer, image_processor, processor
    
def get_text_inputs(model_name, tokenizer, question, image_tensor, image, processor):
    if "llava_phi" in model_name:
        prompt = f"<|user|>\n<image>\n{question}<|end|>\n<|assistant|>\n"
        text_input = tokenizer(prompt, return_tensors='pt')
        inputs = {**text_input, "pixel_values": image_tensor}
    elif "llava" in model_name:
        prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{question} ASSISTANT:"
        text_input = tokenizer(prompt, return_tensors='pt')
        inputs = {**text_input, "pixel_values": image_tensor}
    elif "llama-3.2" in model_name:
        try:
            question = question[:question.index("?")+1]
        except:
            question = question
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text":f"For the following questions, please answer yes or no directly and do not include any other information:\n\n{question}"}
            ]}
        ]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        )
    elif "instructblip" in model_name:
        inputs = None 

    return inputs
        
        
def mme_forward(model_name, img_path, img_name, text_path, model, tokenizer, image_processor, processor):
    with open(text_path, "r") as f:
        data = [line for line in f.readlines()]
    
    image = Image.open(img_path)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(model.device)
    outputs = []
    for line in data:
        question, answer = line.split("\t")[0], line.split("\t")[-1]
        try:
            inputs = get_text_inputs(model_name, tokenizer, question, image_tensor, image, processor)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            output = model.generate(**inputs, max_new_tokens=3)
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
            elif "llama-3.2" in model_name:
                prediction = processor.decode(output[0])
                prediction = prediction[prediction.index("assistant<|end_header_id|>")+ len("assistant<|end_header_id|>"):].strip("\n").strip(" ")
                # if "yes" not in prediction.lower() and "no" not in prediction.lower():
                #     prediction = answer
            outputs.append("\t".join([img_name, question.strip("\n"), answer.strip("\n"), prediction.strip("\n")]))
        except:
            outputs.append["\t".join([img_name, question.strip("\n"), answer.strip("\n"), answer.strip("\n")])]
        print(outputs[-1])
    return outputs
    
def main(args):
    
    model, tokenizer, image_processor, processor = load_model(args)
    
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
                output = mme_forward(args.model_name, img_path, img_name, text_path, model, tokenizer, image_processor, processor)
                outputs.extend(output)
        else:
            for img_name in os.listdir(path):
                if ".png" not in img_name and ".jpg" not in img_name: continue
                img_path = os.path.join(path, img_name)
                text_path = os.path.join(path, f"{img_name.split('.')[0]}.txt")
                output = mme_forward(args.model_name, img_path, img_name, text_path, model, tokenizer, image_processor, processor)
                output = mme_forward(args.model_name, img_path, img_name, text_path, model, tokenizer, image_processor, processor)
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
CUDA_VISIBLE_DEVICES=0 python ./eval/eval_mme.py \
    --model_name llava_phi \
    --model_path ./models/final_ft_10_epochs_lr2e-05_llava-phi_full \
    --mme_dir ./eval/MME_Benchmark_release_version/ \
    --output_dir ./eval/eval_tool/llava_phi_ft

CUDA_VISIBLE_DEVICES=0 python ./eval/eval_mme.py \
    --model_name llava_phi \
    --model_path ./models/final_ft_10_epochs_lr2e-05_llava-phi_full \
    --ckpt_path ./models/final_ft_10_epochs_lr2e-05_llava-phi_full/ga_0.0001_forget5_5/checkpoint.pt \
    --mme_dir ./eval/MME_Benchmark_release_version/ \
    --output_dir ./eval/eval_tool/ga_0.0001_forget5_5 --use_lora True

CUDA_VISIBLE_DEVICES=3 python ./eval/eval_mme.py \
    --model_name llava_phi \
    --model_path ./models/final_ft_10_epochs_lr2e-05_llava-phi_full \
    --ckpt_path ./models/final_ft_10_epochs_lr2e-05_llava-phi_full/gd_0.0001_forget5_5/checkpoint.pt \
    --mme_dir ./eval/MME_Benchmark_release_version/ \
    --output_dir ./eval/eval_tool/gd_0.0001_forget5_5 --use_lora True

CUDA_VISIBLE_DEVICES=0 python ./eval/eval_mme.py \
    --model_name llava_phi \
    --model_path ./models/final_ft_10_epochs_lr2e-05_llava-phi_full \
    --ckpt_path ./models/final_ft_10_epochs_lr2e-05_llava-phi_full/icd_0.0001_forget5_5/checkpoint.pt \
    --mme_dir ./eval/MME_Benchmark_release_version/ \
    --output_dir ./eval/eval_tool/icd_0.0001_forget5_5 --use_lora True

CUDA_VISIBLE_DEVICES=1 python ./eval/eval_mme.py \
    --model_name llava_phi \
    --model_path ./models/final_ft_10_epochs_lr2e-05_llava-phi_full \
    --ckpt_path ./models/final_ft_10_epochs_lr2e-05_llava-phi_full/kl_0.0001_forget5_5/checkpoint.pt \
    --mme_dir ./eval/MME_Benchmark_release_version/ \
    --output_dir ./eval/eval_tool/kl_0.0001_forget5_5 --use_lora True

CUDA_VISIBLE_DEVICES=2 python ./eval/eval_mme.py \
    --model_name llava_phi \
    --model_path ./models/final_ft_10_epochs_lr2e-05_llava-phi_full \
    --ckpt_path ./models/final_ft_10_epochs_lr2e-05_llava-phi_full/idk_0.0003_forget5_5/checkpoint.pt \
    --mme_dir ./eval/MME_Benchmark_release_version/ \
    --output_dir ./eval/eval_tool/idk_0.0003_forget5_5 --use_lora True





CUDA_VISIBLE_DEVICES=1 python ./eval/eval_mme.py \
    --model_name llama-3.2 \
    --model_path meta-llama/Llama-3.2-11B-Vision-Instruct \
    --mme_dir ./eval/MME_Benchmark_release_version/ \
    --output_dir ./eval/eval_tool/llama-3.2_pt

CUDA_VISIBLE_DEVICES=2 python ./eval/eval_mme.py \
    --model_name llama-3.2 \
    --model_path ./models/final_ft_10_epochs_lr2e-05_llama-3.2-vision_full/step_600 \
    --mme_dir ./eval/MME_Benchmark_release_version/ \
    --output_dir ./eval/eval_tool/llama-3.2_ft

CUDA_VISIBLE_DEVICES=3 python ./eval/eval_mme.py \
    --model_name llama-3.2 \
    --model_path ./models/final_ft_10_epochs_lr2e-05_llama-3.2-vision_full \
    --mme_dir ./eval/MME_Benchmark_release_version/ \
    --ckpt_path ./models/final_ft_10_epochs_lr2e-05_llama-3.2-vision_full/icd_3e-05_forget5_5/checkpoint.pt \
    --output_dir ./eval/eval_tool/icd_3e-05_forget5_5 --use_lora True

CUDA_VISIBLE_DEVICES=0 python ./eval/eval_mme.py \
    --model_name llama-3.2 \
    --model_path ./models/final_ft_10_epochs_lr2e-05_llama-3.2-vision_full \
    --mme_dir ./eval/MME_Benchmark_release_version/ \
    --ckpt_path ./models/final_ft_10_epochs_lr2e-05_llama-3.2-vision_full/gd_2e-05_forget5_5/checkpoint.pt \
    --output_dir ./eval/eval_tool/gd_2e-05_forget5_5 --use_lora True

CUDA_VISIBLE_DEVICES=1 python ./eval/eval_mme.py \
    --model_name llama-3.2 \
    --model_path ./models/final_ft_10_epochs_lr2e-05_llama-3.2-vision_full \
    --mme_dir ./eval/MME_Benchmark_release_version/ \
    --ckpt_path ./models/final_ft_10_epochs_lr2e-05_llama-3.2-vision_full/idk_0.0003_forget5_5/checkpoint.pt \
    --output_dir ./eval/eval_tool/idk_0.0003_forget5_5 --use_lora True

CUDA_VISIBLE_DEVICES=2 python ./eval/eval_mme.py \
    --model_name llama-3.2 \
    --model_path ./models/final_ft_10_epochs_lr2e-05_llama-3.2-vision_full \
    --mme_dir ./eval/MME_Benchmark_release_version/ \
    --ckpt_path ./models/final_ft_10_epochs_lr2e-05_llama-3.2-vision_full/kl_6e-05_forget5_5/checkpoint.pt \
    --output_dir ./eval/eval_tool/kl_6e-05_forget5_5 --use_lora True

python ./eval/eval_tool/calculation.py \
    --results_dir eval/eval_tool/gd_2e-05_forget5_5/gd_2e-05_forget5_5

0 2 3 5 6
"""



