import os
import json
import pandas as pd
from api import GeminiEvaluator, GPTEvaluator, system_message

from collections import defaultdict
from tqdm import tqdm

import glob

message = """I will now provide you with the caption of a fictitious entity image and some Q&A pairs based on this guide. I need you to complete two tasks: 1. help me Paraphrase the correct answer to ensure diversity of answers; 2. Perturbed the paraphrased answer to ensure that the overall structure of the answer remains unchanged and that the perturbed answer contains factual errors. 

Caption:
{caption}

Question Answer Pairs:
{qa}

Paraphrased Answer:

Perturbed Answer (Note that the perturbed answer must be very obviously wrong compared to the correct answer. I want you to generate 5 perturbed answers for each paraphrased answers):"""


if __name__ == "__main__":
    
    agent = GeminiEvaluator(api_key="AIzaSyBkgnk3SwtnaJO3-t278N6MJwaZqKIpWF0")
    
    # agent = GPTEvaluator(api_key="0836470ebf0e4368afc31bd03a7d26f9")
    root = "./dataset/"
    
    pbar = tqdm(total=50)
    
    splits = ['exp2', 'exp3', 'exp4']

    for exp in splits:
        path = os.path.join(root, exp)
        forget_path_list = glob.glob(f"{path}/forget*")
        real_path_list = glob.glob(f"{path}/real*")
        
        for item in forget_path_list:
            if "perturbed" in item: continue
            forget_path = item

        for item in real_path_list:
            if "perturbed" in item: continue
            real_path = item


        forget_name = forget_path.split("/")[-1].split(".")[0].strip(" ")
        real_name = real_path.split("/")[-1].split(".")[0].strip(" ")

        with open(forget_path, "r") as f:
            forget_data = [json.loads(line) for line in f.readlines()]

        with open(real_path, "r") as f:
            real_data = [json.loads(line) for line in f.readlines()]

        cnt = 0
        with open(os.path.join(path, f"{forget_name}_perturbed.json"), "w") as f:
            for line in tqdm(forget_data):
                cnt += 1
                # if cnt <= 137: continue
                qa = f"Question: {line['question']}\nOrigin Answer: {line['answer']}"
                question = {
                    "prompted_system_content": "",
                    "prompted_content": message.format(caption=line['caption'], qa=qa),
                    "image_list": None,
                }
            
                response = agent.generate_answer(question)
                outputs = {
                    "response": response['prediction']
                }
                print(outputs)

                f.write(f"{json.dumps(outputs)}\n")
                f.flush()

    
                
                
            
        

# import os
# from openai import AzureOpenAI

# from openai import AzureOpenAI

# api_base = "https://yingzi-west.openai.azure.com/"
# api_key= "0836470ebf0e4368afc31bd03a7d26f9"
# # deployment_name = "gpt-4-vision-preview"
# deployment_name = "gpt-35-turbo"
# api_version = "2024-02-01"

# client = AzureOpenAI(
#     api_key=api_key,
#     api_version=api_version,
#     base_url=f"{api_base}/openai/deployments/{deployment_name}"
# )

# response = client.chat.completions.create(
#     model=deployment_name,
#     messages=[
#         { "role": "system", "content": "You are a helpful assistant." },
#         { "role": "user", "content": [
#             {
#                 "type": "text",
#                 "text": "Describe this picture:"
#             },
#             # {
#             #     "type": "image_url",
#             #     "image_url": {
#             #         "url": "https://th.bing.com/th/id/OIP.avb9nDfw3kq7NOoP0grM4wHaEK?rs=1&pid=ImgDetMain"
#             #     }
#             # }
#         ] }
#     ],
#     max_tokens=2000
# )

# print(response)