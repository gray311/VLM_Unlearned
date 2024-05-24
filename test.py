import os
import json
import pandas as pd
from api import GeminiEvaluator, GPTEvaluator, system_message

from collections import defaultdict
from tqdm import tqdm



if __name__ == "__main__":
    
    file_path = "/home/scratch.chaoweix_nvresearch/av/VLM_Unlearned/dataset/vlm_unlearned_examples.json"
    # agent = GeminiEvaluator(api_key="AIzaSyCDsKWXyZTLhU7vHL-f4Ozlr7FY3HJyWKA")
    
    agent = GPTEvaluator(api_key="d4621a9612bf4699a832a974b5b761ed")
    with open(file_path, "r") as f:
        examples = [json.loads(line) for line in f.readlines()]

    for line in examples:
        question = {
            "prompted_system_content": "",
            "prompted_content": system_message['illustration'],
            "image_list": line['image_list']
        }
        response = agent.generate_answer(question)
        print(response)
        break