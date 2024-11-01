import os
import json
import random

random.seed(233)

prompts = [
    "You are an expert in image analysis. Please analyze this image and determine whether it is a natural image or a modified one. Provide a label based on your assessment.",
    "As a professional in digital forensics, your task is to examine this image and classify it as either 'natural' or 'modified.'",
    "Please act as a specialist in computer vision and assess whether this image has been altered. Label it as 'natural' or 'modified' accordingly.",
    "You are an expert in identifying image modifications. Review the given image and decide if it is 'natural' or 'modified.' Provide a label.",
    "As a specialist in photo editing detection, your task is to determine whether this image is original or modified. Please classify it and label it appropriately."
]

natural_responses = [
    "This image appears to be a natural image.",
    "Based on my analysis, the image is natural.",
    "This image is natural.",
    "The image is a natural image.",
    "This image is original (natural)."
]

# Modified responses list
modified_responses = [
    "This image seems to be a modified image.",
    "The image is modified.",
    "This image has been modified.",
    "The image is a modified image.",
    "This image has been modified."
]

"""
{
    image_path:
    question:
    answer:
    label:
}
"""


root = "./dataset"
split = "val"
file_path = os.path.join(root, split)

with open(f"./{split}.json", "w") as f:
    for cate in ["fakes", "reals"]:
        cate_path = os.path.join(file_path, cate)
        for image_name in os.listdir(cate_path):
            example = {}
            question = random.choice(prompts)
            if cate == "fakes":
                answer = random.choice(modified_responses)
                label = "modified"
            else:
                answer = random.choice(natural_responses)
                label = "natural"

            image_path = os.path.join(cate_path, image_name)
            example['question'] = question
            example['answer'] = answer
            example['image_path'] = image_path
            example['label'] = label
            f.write(f"{json.dumps(example)}\n")

        

