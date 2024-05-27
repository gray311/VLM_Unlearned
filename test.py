# import os
# import json
# import pandas as pd
# from api import GeminiEvaluator, GPTEvaluator, system_message

# from collections import defaultdict
# from tqdm import tqdm



# if __name__ == "__main__":
    
#     # agent = GeminiEvaluator(api_key="AIzaSyCDsKWXyZTLhU7vHL-f4Ozlr7FY3HJyWKA")
    
#     agent = GPTEvaluator(api_key="0836470ebf0e4368afc31bd03a7d26f9")
#     root = "/home/scratch.chaoweix_nvresearch/av/VLM_Unlearned/dataset/images/real"
    
#     pbar = tqdm(total=50)
#     with open("/home/scratch.chaoweix_nvresearch/av/VLM_Unlearned/dataset/raw_real.json", "a+") as f:
#         for label in os.listdir(root):
#             path = os.path.join(root, label)
#             for img_name in os.listdir(path):
#                 if ".png" not in img_name: continue
#                 image_path = os.path.join(path, img_name)
#                 question = {
#                     "prompted_system_content": "",
#                     "prompted_content": system_message['real'],
#                     "image_list": [image_path]
#                 }
#                 response = agent.generate_answer(question)
#                 outputs = {
#                     "image_name": img_name,
#                     "image_path": image_path,
#                     "response": response['prediction']
#                 }

#                 f.write(f"{json.dumps(outputs)}\n")
#                 f.flush()
#                 pbar.update(1)
                
                
            
        

import os
from openai import AzureOpenAI

from openai import AzureOpenAI

api_base = "https://yingzi-west.openai.azure.com/"
api_key= "0836470ebf0e4368afc31bd03a7d26f9"
# deployment_name = "gpt-4-vision-preview"
deployment_name = "gpt-35-turbo"
api_version = "2024-02-01"

client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    base_url=f"{api_base}/openai/deployments/{deployment_name}"
)

response = client.chat.completions.create(
    model=deployment_name,
    messages=[
        { "role": "system", "content": "You are a helpful assistant." },
        { "role": "user", "content": [
            {
                "type": "text",
                "text": "Describe this picture:"
            },
            # {
            #     "type": "image_url",
            #     "image_url": {
            #         "url": "https://th.bing.com/th/id/OIP.avb9nDfw3kq7NOoP0grM4wHaEK?rs=1&pid=ImgDetMain"
            #     }
            # }
        ] }
    ],
    max_tokens=2000
)

print(response)