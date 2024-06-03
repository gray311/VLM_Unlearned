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
from peft import LoraConfig, get_peft_model

random.seed(233)

caption_prompt = {
    "normal": ['Please provide a detailed description of the image.', 'Can you describe the image thoroughly?', 'Give a comprehensive description of the image.', 'Please explain what is happening in the image in detail.', 'Describe all the elements present in the image.', 'Provide a detailed narrative of the scene depicted in the image.', 'What are the key features in the image? Please describe them.', "Please give an in-depth description of the image's content.", "Can you explain the image's details and context?", 'Describe the image, including all noticeable aspects.', 'Please elaborate on the visual details in the image.', 'What do you see in the image? Provide a detailed description.', 'Can you break down the elements of the image for me?', 'Please describe the image scene comprehensively.', 'Give a full description of everything visible in the image.', 'Describe the main subjects and background in the image.', 'Can you detail the visual composition of the image?', 'Please describe the setting and characters in the image.', "Explain the image's details, including colors, shapes, and objects.", "Can you describe the image as if I couldn't see it?"],
    "attribute": ['Can you describe the {attribute} of the entity in the image in detail?', 'Please explain the {attribute} of the entity in the image thoroughly.', "Give a comprehensive description of the entity's {attribute} in the image.", 'Please describe the {attribute} of the entity depicted in the image.', "Provide a detailed narrative of the entity's {attribute} in the image.", "What are the key features of the entity's {attribute} in the image? Please describe them.", "Please give an in-depth description of the entity's {attribute} in the image.", "Can you explain the entity's {attribute} details and context in the image?", "Describe the entity's {attribute} in the image, including all noticeable aspects.", "Please elaborate on the visual details of the entity's {attribute} in the image.", "What do you see regarding the entity's {attribute} in the image? Provide a detailed description.", 'Can you break down the {attribute} of the entity in the image for me?', "Please describe the entity's {attribute} in the image comprehensively.", "Give a full description of the entity's {attribute} visible in the image.", 'Describe the main subjects and their {attribute} in the image.']
}


def main():
    forget_files = ["forget5.json", "forget10.json", "forget5_random.json", "forget_attribute.json"]
    
    model_path = "models/vlm_unlearned_ft_llava_v1.6_vicuna_7b" 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlavaForConditionalGeneration.from_pretrained(model_path, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

    model.half().to("cuda:3")
    model.eval()
    
    for i, file in enumerate(forget_files):
        split = f"exp{str(i+1)}"
        file = os.path.join("./dataset", split, file)
        # file = "./dataset/exp1/forget5_perturbed.json"
        # split = "exp1"
        loss_type = "icd"
        file_name = file.split("/")[-1].split(".")[0].strip(" ")
        
        with open(file, "r") as f:
            data = [json.loads(line) for line in f.readlines()]
            
        # random.shuffle(data)
        print(
            f"Full dataset length (only include fictitious examples): {len(data)}."
        )
        eval_data = data
        print(
            f"Subset length of the full dataset for evaluation: {len(eval_data)}."
        )
        
        nlls = []
        with open(f"./dataset/{split}/icd_caption.json", "w") as f:
            for j, line in enumerate(tqdm(eval_data)):
                image_path = line['image_path']
                image = Image.open(image_path)
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(model.device)
                
                # question, answer = line['question'], line['answer']
                if line['attribute'] == "normal":
                    random.shuffle(caption_prompt['normal'])
                    question = caption_prompt["normal"][0]
                else:
                    random.shuffle(caption_prompt['attribute'])
                    question = caption_prompt["attribute"][0].format(attribute=line['attribute'])
                
                prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{question} ASSISTANT:"
                text_input = tokenizer(prompt, return_tensors='pt')
                text_input = {k: v.to(model.device) for k, v in text_input.items()}
                inputs = {**text_input, "pixel_values": image_tensor}
                
                output = model.generate(
                    **inputs, 
                    max_new_tokens=512,
                    num_beams=5,
                    num_return_sequences=3,
                )
                
                outputs = []
                for k in range(output.shape[0]):
                    prediction = tokenizer.decode(output[k], skip_special_tokens=True)
                    prediction = prediction[prediction.rfind("ASSISTANT:") + len("ASSISTANT:"):].strip(" ")
                    outputs.append(prediction)

                outputs = {
                        "question": question,
                        "caption": outputs,
                        "image_path": line['image_path']
                    }
                
                print(outputs)
                f.write(f"{json.dumps(outputs)}\n")
        
    
if __name__ == "__main__":
    main()
        
