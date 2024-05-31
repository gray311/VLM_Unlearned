"""OpenAI GPT Evaluator"""

import openai
from openai import AzureOpenAI, OpenAI
import requests
import json
from tqdm import tqdm
import random
import time
import pdb
from api.utils import encode_image_base64

class GPTEvaluator:
    def __init__(self, api_key, api_base="https://yingzi-west.openai.azure.com" , model='gpt-4o-2024-05-13', max_tokens=256, temperature=0.2):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = AzureOpenAI(
            api_key=api_key,  
            api_version="2023-12-01-preview",
            base_url=f"{api_base}/openai/deployments/{model}"
        )
        
        
    def prepare_inputs(self, question):
        image_list = question.get("image_list")
        messages = [{
            "role": "system",
            "content": question["prompted_system_content"]
        }]

        if image_list:
            user_message = {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": question["prompted_content"]
                },]}
            for image_path in image_list:
                base64_image = encode_image_base64(image_path) # max_size = 512
                user_message["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "auto"
                    },},)
            messages.append(user_message)
        else:
            messages.append({
                "role": "user",
                "content": question["prompted_content"]
            }, )

        return messages

    def generate_response(self, question):
        message = self.prepare_inputs(question)
        response = ""
        response_ = None
        i = 0
        MAX_RETRY = 100
            
        while i < MAX_RETRY:
            try:
                response_ = self.client.chat.completions.create(
                    model=self.model,
                    messages=message,
                    temperature = self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                if response_ == None:
                    raise Exception("Chatgpt ret error.")
                response = response_.choices[0].message.content.strip()
            except KeyboardInterrupt:
                raise Exception("Terminated by user.")
            except Exception:
                print(response_)
                error=""
                try:
                    error=response_["error"]["message"].split("(request id:")[0].strip()
                    print(error)
                    print(response_.json())
                except:
                    pass
                i += 1
                time.sleep(1 + i / 10)
                if i == 1 or i % 10 == 0:
                    if error.startswith("This model's maximum context length") or error.startswith("Your input image may contain"):
                        response = ""
                        feedback = error
                        return response, message, feedback
                    print(f"Retry {i} times...")
            else:
                break
        if i >= MAX_RETRY:
            raise Exception("Failed to generate response.")
        return response, message, None

    def generate_answer(self, question):
        response, message, feedback = self.generate_response(question)
        if not isinstance(message[1]["content"], str):
            for i in range(len(message[1]["content"])):
                if message[1]["content"][i]["type"] == "image_url":
                    message[1]["content"][i]["image_url"]["url"] = message[1]["content"][i]["image_url"]["url"][:64]+"..."
        question["input_message"] = message
        question["prediction"] = response
        if feedback:
            question["feedback"] = feedback
        question.pop("prompted_content")
        question.pop("prompted_system_content")
        return question