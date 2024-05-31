import os
import json
from typing import Dict, Optional, Sequence, List
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
import datasets
from PIL import Image
import transformers
import glob

from utils import get_model_identifiers_from_yaml

def preprocess_v1(tokenizer, input_ids, conversation, roles, ignore_index=-100):
    target = input_ids.clone()
    total_len = int(target.ne(tokenizer.pad_token_id).sum())
    cur_len = 1
    target[:, :cur_len] = ignore_index
    instruction = conversation.split(roles[1])[0].strip(" ")
    instruction_len = len(tokenizer(instruction + roles[1])['input_ids']) - 2
    target[:, cur_len : cur_len + instruction_len] = ignore_index
    # target[target == -100] = 0
    return target
    
    
class MMDatasetQA(Dataset):
    def __init__(self, config, tokenizer, image_processor, max_length=512, split=None):
        super(MMDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        
        try:
            with open(config.data_path, "r") as f:
                self.data = json.load(f)
        except:
            with open(config.data_path, "r") as f:
                self.data = [json.loads(line) for line in f.readlines()]
                
        self.model_configs = get_model_identifiers_from_yaml(config.model_family)
        
        self.samples = []
        if "full" in config.data_path:
            for line in self.data:
                qa_list = line['question_and_answer']
                for qa in qa_list:
                    qa.update(image_path=line['image_path'])
                    if split == "attribute" and qa['attribute']:
                        self.samples.append(qa)
                    else:
                        self.samples.append(qa)
        else:
            for line in self.data:
                self.samples.append(
                    {
                        "image_path": line['image_path'], 
                        "q": line['question'], 
                        "a": line['answer'], 
                        "attribute": line['attribute'], 
                    }
                )
        
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
    

class MMForgetDatasetQA(Dataset):
    def __init__(self, config, tokenizer, image_processor, max_length=512, split=None):
        super(MMForgetDatasetQA, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        
        forget_path_list = glob.glob(f"{config.data_path}/{config.split}/forget*")
        retain_path_list = glob.glob(f"{config.data_path}/{config.split}/retain*")
        real_path_list = glob.glob(f"{config.data_path}/{config.split}/real*")

        for item in forget_path_list:
            if "perturbed" in item: continue
            forget_path = item

        for item in retain_path_list:
            if "perturbed" in item: continue
            retain_path = item

        for item in real_path_list:
            if "perturbed" in item: continue
            real_path = item
        
        print(forget_path)
        
        with open(forget_path, "r") as f:
            self.forget_data = [json.loads(line) for line in f.readlines()]
        with open(retain_path, "r") as f:
            self.retain_data = [json.loads(line) for line in f.readlines()]
        with open(real_path, "r") as f:
            self.real_data = [json.loads(line) for line in f.readlines()]
                
        self.model_configs = get_model_identifiers_from_yaml(config.model_family)
        
        if config.forget_loss == "idk":
            self.split1, self.split2 = "idk", "retain"
            self.idontknowfile = "dataset/idontknow.json"
            with open(self.idontknowfile, "r") as f:
                self.idk = json.load(f)
        else:
            self.split1, self.split2 = "forget", "retain"
            
        print(
            f"There are {len(self.forget_data)} QA pairs of forgetting dataset!\n",
            f"There are {len(self.retain_data)} QA pairs of forgetting dataset!\n",
            f"There are {len(self.real_data)} QA pairs of forgetting dataset!\n",
        )
        
    
    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = [] # (forget, retain)
        for data_type in [self.split1, self.split2]:
            data = self.retain_data if data_type == "retain" else self.forget_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)

            image_path = data[idx]['image_path']
            question = data[idx]['question']
            answer = data[idx]['answer']
            
            if data_type == "idk":
                attribute = data[idx]['attribute']
                idk = self.idk[attribute]
                rand_pos = torch.randint(0, len(idk), (1,)).item()
                answer = idk[rand_pos].strip(" ")

            image_tensor = self.image_processor.preprocess(Image.open(image_path), return_tensors='pt')['pixel_values']
            system_message = self.model_configs['system_tag']
            if "llava" in self.config.model_family:
                roles = [self.model_configs['question_start_tag'], self.model_configs['answer_tag']]
                conversation = system_message + roles[0] + "<image>\n" + question + roles[1] + answer
            text_input = self.tokenizer(conversation, max_length=self.max_length, truncation=True, return_tensors="pt")
            labels = preprocess_v1(self.tokenizer, text_input['input_ids'], conversation, roles)
            rets.append({**text_input, "labels": labels, "pixel_values": image_tensor})
            
        return rets
    
 
    
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

def pad_to_length(tensor, target_length, pad_value):
    padding_size = target_length - tensor.size(1)
    padding_tensor = torch.full((tensor.size(0), padding_size), pad_value)
    return torch.cat((tensor, padding_tensor), dim=1)

@dataclass
class custom_data_collator_forget(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        forget_instances, retain_instances = [instance[0] for instance in instances], [instance[1] for instance in instances]
        forget_input_ids, forget_labels = tuple([sample[key][0] for sample in forget_instances] for key in ("input_ids", "labels"))
        retain_input_ids, retain_labels = tuple([sample[key][0] for sample in retain_instances] for key in ("input_ids", "labels"))

        input_ids_max_length = -1
        for input_ids in forget_input_ids:
            input_ids_max_length = max(input_ids_max_length, input_ids.shape[-1])
        for input_ids in retain_input_ids:
            input_ids_max_length = max(input_ids_max_length, input_ids.shape[-1])
        
        labels_max_length = -1
        for labels in forget_labels:
            labels_max_length = max(labels_max_length, labels.shape[-1])
        for labels in retain_labels:
            labels_max_length = max(labels_max_length, labels.shape[-1])

        rets = []
        for data_type in ["forget", "retain"]:
            samples = forget_instances if data_type == "forget" else retain_instances
            input_ids, labels = tuple([sample[key][0] for sample in samples] for key in ("input_ids", "labels"))

            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                    batch_first=True,
                                                    padding_value=-100)
            input_ids = pad_to_length(input_ids, input_ids_max_length, self.tokenizer.pad_token_id)
            labels = pad_to_length(labels, labels_max_length, -100)

            input_ids = input_ids[:, :self.tokenizer.model_max_length]
            labels = labels[:, :self.tokenizer.model_max_length]
        
            batch = dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )
            if 'pixel_values' in samples[0]:
                pixel_values = [samples['pixel_values'].squeeze(0) for samples in samples]
                if all(x is not None and x.shape == pixel_values[0].shape for x in pixel_values):
                    batch['pixel_values'] = torch.stack(pixel_values)
                else:
                    batch['pixel_values'] = pixel_values

            rets.append(batch)
                
        return rets