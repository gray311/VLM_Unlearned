import os
import json
from typing import Dict, Optional, Sequence, List
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
import datasets
from PIL import Image
import transformers

from utils import get_model_identifiers_from_yaml

def preprocess_v1(tokenizer, input_ids, conversation, roles, ignore_index=-100):
    target = input_ids.clone()
    total_len = int(target.ne(tokenizer.pad_token_id).sum())
    cur_len = 1
    target[:, :cur_len] = ignore_index
    instruction = conversation.split(roles[1])[0].strip(" ")
    instruction_len = len(tokenizer(instruction + roles[1])['input_ids']) - 2
    target[:, cur_len : cur_len + instruction_len] = ignore_index
    target[target==-100] = 0
    return target
    
    
class MultiModalDatasetQA(Dataset):
    def __init__(self, config, tokenizer, image_processor, max_length=512, split=None):
        super(MultiModalDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        
        self.data = json.load(open(config.data_path, "r"))
        self.model_configs = get_model_identifiers_from_yaml(config.model_family)
        
        self.samples = []
        for line in self.data:
            qa_list = line['question_and_answer']
            for qa in qa_list:
                qa.update(image_path=line['image_path'])
                if split == "attribute" and qa['attribute']:
                    self.samples.append(qa)
                else:
                    self.samples.append(qa)
        
        print(
            f"There are {len(self.samples)} QA pairs for fine-tuning!"
        )
        
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path = self.samples[idx]['image_path']
        question = self.samples[idx]['q']
        answer = self.samples[idx]['a']
        image_tensor = self.image_processor.preprocess(Image.open(image_path), return_tensors='pt')['pixel_values']
        system_message = self.model_configs['system_tag']
        roles = [self.model_configs['question_start_tag'], self.model_configs['answer_tag']]
        conversation = system_message + roles[0] + "<image>\n" + question + roles[1] + answer
        text_input = self.tokenizer(conversation, max_length=self.max_length, truncation=True, return_tensors="pt")
        labels = preprocess_v1(self.tokenizer, text_input['input_ids'], conversation, roles)

        return {**text_input, "labels": labels, "pixel_values": image_tensor}
                
    
def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks



@dataclass
class custom_data_collator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key][0] for instance in instances] for key in ("input_ids", "labels"))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=-100)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        if 'pixel_values' in instances[0]:
            pixel_values = [instance['pixel_values'].squeeze(0) for instance in instances]
            if all(x is not None and x.shape == pixel_values[0].shape for x in pixel_values):
                batch['pixel_values'] = torch.stack(pixel_values)
            else:
                batch['pixel_values'] = pixel_values
                
        return batch
