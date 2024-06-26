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
from transformers import ( 
    InstructBlipProcessor, 
    InstructBlipForConditionalGeneration
)
from peft import LoraConfig, get_peft_model

random.seed(233)


def main(eval_file):
    if eval_file is None:
        file = "./dataset/exp1/retain95.json"
        split = "exp1"
        loss_type = "ga"
        file_name = file.split("/")[-1].split(".")[0].strip(" ")
        # model_path = "./models/final_ft_LORA_6_epochs_inst_lr0.0001_llava-v1.6-vicuna_full"
        model_path = "models/vlm_unlearned_ft_llava_phi_3_mini"
        if "llava" in model_path:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = LlavaForConditionalGeneration.from_pretrained(model_path, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
            image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

            # target_modules=r'.*language_model.*\.(up_proj|k_proj|linear_2|down_proj|v_proj|q_proj|o_proj|gate_proj|linear_1)'
            # config = LoraConfig(
            #     r=128, 
            #     lora_alpha=256, 
            #     target_modules=target_modules, 
            #     lora_dropout=0.05,
            #     bias="none", 
            #     task_type="CAUSAL_LM"
            # )
            # model = get_peft_model(model, config)
            # checkpoint_path = "./models/vlm_unlearned_ft_llava_v1.6_vicuna_7b/KL_4e-05_exp2_8/checkpoint.pt"
          
            # model_state = torch.load(checkpoint_path)
            # model.load_state_dict(torch.load(checkpoint_path), strict=False)
            # model.merge_and_unload() 
            model.half().to("cuda:0")
            model.eval()

        elif "instructblip" in model_path:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = InstructBlipForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
            image_processor = InstructBlipProcessor.from_pretrained(model_path)

            # target_modules=r'.*language_model.*\.(o|k|q|v|wi_0|wi_1|wo)'
            # config = LoraConfig(
            #     r=cfg.LoRA.r, 
            #     lora_alpha=cfg.LoRA.alpha, 
            #     target_modules=target_modules, 
            #     lora_dropout=cfg.LoRA.dropout,
            #     bias="none", 
            #     task_type="CAUSAL_LM"
            # )
            # model = get_peft_model(model, config)
            # checkpoint_path = "./models/vlm_unlearned_ft_llava_v1.6_vicuna_7b/KL_2e-05_exp3_8/step_24/checkpoint.pt"
            # model_state = torch.load(checkpoint_path)
            # model.load_state_dict(torch.load(checkpoint_path), strict=False)
            # model.merge_and_unload() 

            model.half().to("cuda:0")
            model.eval()

        with open(file, "r") as f:
            data = [json.loads(line) for line in f.readlines()]

        # random.shuffle(data)
        print(
            f"Full dataset length (only include fictitious examples): {len(data)}."
        )
        eval_data = data[:400]
        print(
            f"Subset length of the full dataset for evaluation: {len(eval_data)}."
        )

        nlls = []
        with open(f"./outputs/{split}_{loss_type}_{file_name}_results.json", "w") as f:
            for line in tqdm(eval_data):
                image_path = line['image_path']
                image = Image.open(image_path)
                question, answer = line['question'], line['answer']
                if "llava" in model_path:
                    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(model.device)
                    prompt = f"<|user|>\n<image>\n{question}<|end|>\n<|assistant|>\n" ### LLaVA-Phi
                    text_input = tokenizer(prompt, return_tensors='pt')
                    text_input = {k: v.to(model.device) for k, v in text_input.items()}
                    inputs = {**text_input, "pixel_values": image_tensor}
                    output = model.generate(**inputs, max_new_tokens=40)
                    prediction = tokenizer.decode(output[0])
                    prediction = prediction[prediction.find("<|assistant|>") + len("<|assistant|>"): ].strip(" ")

                elif "instructblip" in model_path:
                    inputs = image_processor(images=image, text=question, return_tensors="pt").to(model.device)
                    prompt = f"Question: {question} Answer:"
                    text_inputs = tokenizer(prompt, return_tensors="pt")
                    inputs.update(input_ids=text_inputs['input_ids'])
                    inputs.update(attention_mask=text_inputs['attention_mask'])
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    outputs = model.generate(
                        **inputs, 
                        do_sample=False,
                        num_beams=5,
                        max_length=64,
                        min_length=1,
                        top_p=0.9,
                        repetition_penalty=1.5,
                        length_penalty=1.0,
                        temperature=1,
                    )
                    prediction = image_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()


                outputs = {
                        "question": question,
                        "answer": answer,
                        "prediction": prediction 
                    }

                print(outputs)
                f.write(f"{json.dumps(outputs)}\n")   
    else:
        with open(eval_file, "r") as f:
            data = [json.loads(line) for line in f.readlines()]

        rougeL_list = []
        for line in data:
            pred, gt = line['prediction'], line['answer']
            pred = pred[:pred.find(".")].strip("")
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(gt, pred)
            rougeL_list.append(rouge_scores['rougeL'].precision)

        print(
            f"Avg RougeL scores: {sum(rougeL_list) / len(rougeL_list)}"
        )



if __name__ == "__main__":
    eval_file = "outputs/exp1_ga_retain95_results.json"
    # eval_file = None
    main(eval_file)
