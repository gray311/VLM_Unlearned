import os
import sys
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

data_split = json.load(open("./dataset/split.json", "r"))

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

def pad_sequence(sequences, padding_side='right', padding_value=0, max_len=None):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

def pad_qformer_input_ids(input_ids_list, pad_token_id, max_length=50):
    padded_input_ids_list = []
    for input_ids in input_ids_list:
        if len(input_ids) > max_length:
            padded_input_ids = input_ids[:max_length]
        else:
            pad_tensor = [pad_token_id] * (max_length - len(input_ids))
            pad_tensor = torch.tensor(pad_tensor)
            padded_input_ids = torch.cat([input_ids, pad_tensor])
        padded_input_ids_list.append(padded_input_ids)
    
    padded_input_ids_list = [tensor.tolist() for tensor in padded_input_ids_list]
    padded_input_ids_tensor = torch.tensor(padded_input_ids_list)
    return padded_input_ids_tensor
    
    
class MMDatasetQA(Dataset):
    def __init__(self, config, tokenizer, image_processor, data_path=None, max_length=512, split=None, processor=None, question_key="q", answer_key="a"):
        super(MMDatasetQA, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.processor = processor
        self.max_length = max_length

        self.question_key = question_key
        self.answer_key = answer_key
        
        self.data_path = data_path if data_path is not None else config.data_path
        try:
            with open(self.data_path, "r") as f:
                self.data = json.load(f)
        except:
            with open(self.data_path, "r") as f:
                self.data = [json.loads(line) for line in f.readlines()]
        
        # self.data = self.data[:200]
        self.model_configs = get_model_identifiers_from_yaml(config.model_family)
        
        self.samples = []
        self.data = self.data[:400] ### choose 400 person for evaluation
        if split is not None:
            if split in data_split.keys():
                self.data = [line for line in self.data if line['unique_id'] in data_split[split]]
            elif split == "retain":
                ignore_index = data_split['forget1'] + data_split['forget5'] + data_split['forget10'] 
                self.data = [line for line in self.data if line['unique_id'] not in ignore_index]
        
        for line in self.data:
            qa_list = line['qa_list']
            for qa in qa_list:
                qa.update(label="human_face")
                qa.update(image_path=line['image_path'])
                question = qa[question_key]
                if isinstance(question, str):
                    question = [question]
                for i, q in enumerate(question):
                    robust_qa = qa.copy()
                    robust_qa['paraphrased_question'] = q
                    self.samples.append(robust_qa)
        

        print(
            f"There are {len(self.samples)} QA pairs for fine-tuning or evaluation!"
        )
        
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path = self.samples[idx]['image_path']
        question = self.samples[idx][self.question_key].capitalize()
        answers = self.samples[idx][self.answer_key]
        category = self.samples[idx]['label']
        if isinstance(answers, str):
            answers = [answers.capitalize()]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []
        pixel_value_list = []
        aspect_ratio_ids_list = []
        aspect_ratio_mask_list = []
        cross_attention_mask_list = []

        
        if "llava" in self.config.model_family:
            image_tensor = self.image_processor.preprocess(Image.open(image_path), return_tensors='pt')['pixel_values']
            for ans in answers:
                system_message = self.model_configs['system_tag']
                roles = [self.model_configs['question_start_tag'], self.model_configs['answer_tag']]
                conversation = system_message + roles[0] + "<image>\n" + question + roles[1] + ans
                text_input = self.tokenizer(conversation, max_length=self.max_length, truncation=True, return_tensors="pt")
                label = preprocess_v1(self.tokenizer, text_input['input_ids'], conversation, roles)
                pad_input_ids_list.append(text_input['input_ids'][0])
                pad_attention_mask_list.append(text_input['attention_mask'][0])
                label_list.append(label[0])
                pixel_value_list.append(image_tensor)

        elif "instructblip" in self.config.model_family:
            pad_qformer_input_ids_list = []
            pad_qformer_attention_mask_list = []
            for ans in answers:
                inputs = self.image_processor(images=Image.open(image_path), text=question, return_tensors="pt")
                system_message = self.model_configs['system_tag']
                roles = [self.model_configs['question_start_tag'], self.model_configs['answer_tag']]
                conversation = system_message + roles[0] + question + roles[1] + ans
                text_input = self.tokenizer(conversation, max_length=self.max_length, truncation=True, return_tensors="pt")
                label = preprocess_v1(self.tokenizer, text_input['input_ids'], conversation, roles)
        
                pad_input_ids_list.append(text_input['input_ids'][0])
                pad_attention_mask_list.append(text_input['attention_mask'][0])
                pad_qformer_input_ids_list.append(inputs['qformer_input_ids'][0])
                pad_qformer_attention_mask_list.append(inputs['qformer_attention_mask'][0])
                label_list.append(label[0])
                pixel_value_list.append(inputs['pixel_values'])

        elif "llama-3.2" in self.config.model_family.lower():
            for ans in answers:
                sources = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": ans}
                    ]},
                ]
                roles = [self.model_configs['question_start_tag'], self.model_configs['answer_tag']]
                input_text = self.processor.apply_chat_template(sources)
                inputs = self.processor(Image.open(image_path), input_text, return_tensors="pt")
                image_tensor = inputs['pixel_values']
                labels = preprocess_v1(self.tokenizer, inputs['input_ids'], input_text, roles)

                pad_input_ids_list.append(inputs['input_ids'][0])
                pad_attention_mask_list.append(inputs["attention_mask"][0])
                label_list.append(labels[0])
                pixel_value_list.append(inputs['pixel_values'])
                aspect_ratio_ids_list.append(inputs['aspect_ratio_ids'])
                aspect_ratio_mask_list.append(inputs['aspect_ratio_mask'])
                cross_attention_mask_list.append(inputs['cross_attention_mask'][0])
   
            input_ids = pad_sequence(
                pad_input_ids_list, padding_side='right', padding_value=self.tokenizer.pad_token_id
            )
            attention_mask = pad_sequence(
                pad_attention_mask_list, padding_side='right', padding_value=self.tokenizer.pad_token_id
            )
            labels = pad_sequence(
                label_list, padding_side='right', padding_value=-100
            ) 
            pixel_values = torch.stack(pixel_value_list)
            aspect_ratio_ids = pad_sequence(
                aspect_ratio_ids_list, padding_side='right', padding_value=0
            )
            aspect_ratio_mask = pad_sequence(
                aspect_ratio_mask_list, padding_side='right', padding_value=0
            )
            cross_attention_mask = pad_sequence(
                cross_attention_mask_list, padding_side='right', padding_value=0
            )

            ret =  dict(
                input_ids=input_ids.squeeze(0),
                pixel_values=pixel_values.squeeze(1),
                aspect_ratio_mask=aspect_ratio_mask.squeeze(1),
                aspect_ratio_ids=aspect_ratio_ids.squeeze(1),
                cross_attention_mask=cross_attention_mask,
                attention_mask=attention_mask.squeeze(0),
                labels=labels.squeeze(0),
                category=[category for _ in range(input_ids.shape[0])],
            )

            return ret


        input_ids = torch.nn.utils.rnn.pad_sequence(
            pad_input_ids_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id) 
    
        attention_mask = torch.nn.utils.rnn.pad_sequence(
                pad_attention_mask_list,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)  

        labels = torch.nn.utils.rnn.pad_sequence(
                label_list,
                batch_first=True,
                padding_value=-100)   
        
        pixel_values = torch.stack(pixel_value_list)

        
        if "instructblip" in self.config.model_family:
            qformer_input_ids = pad_qformer_input_ids(pad_qformer_input_ids_list, self.tokenizer.pad_token_id)
            qformer_attention_mask = qformer_input_ids.ne(self.tokenizer.pad_token_id)
            
            return {
                "input_ids": input_ids.squeeze(0), 
                "attention_mask": attention_mask.squeeze(0), 
                "labels": labels.squeeze(0), 
                "qformer_input_ids": qformer_input_ids.squeeze(0),
                "qformer_attention_mask": qformer_attention_mask.squeeze(0),
                "pixel_values": pixel_values.squeeze(0),
                "category": [category for _ in range(input_ids.shape[0])],
            }
         
        else:
            return {
                "input_ids": input_ids.squeeze(0), 
                "attention_mask": attention_mask.squeeze(0), 
                "labels": labels.squeeze(0), 
                "pixel_values": pixel_values.squeeze(0),
                "category": [category for _ in range(input_ids.shape[0])],
            }
    

class MMForgetDatasetQA(Dataset):
    def __init__(self, config, tokenizer, image_processor, max_length=512, split=None, processor=None):
        super(MMForgetDatasetQA, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.processor = processor
        self.max_length = max_length

        self.data_path = config.data_path
        try:
            with open(self.data_path, "r") as f:
                self.data = json.load(f)
        except:
            with open(self.data_path, "r") as f:
                self.data = [json.loads(line) for line in f.readlines()]

        self.data = self.data[:400]

        self.forget_data, self.retain_data = [], []
        if config.split in data_split.keys():
            self.forget_personal_data = [line for line in self.data if line['unique_id'] in data_split[config.split]]
            ignore_index = data_split['forget1'] + data_split['forget5'] + data_split['forget10'] 
            self.retain_personal_data = [line for line in self.data if line['unique_id'] not in ignore_index]
        
            for line in self.forget_personal_data:
                qa_list = line['qa_list']
                for qa in qa_list:
                    qa.update(image_path=line['image_path'])
                    self.forget_data.append(qa)
            
            for line in self.retain_personal_data:
                qa_list = line['qa_list']
                for qa in qa_list:
                    qa.update(image_path=line['image_path'])
                    self.retain_data.append(qa)

        self.model_configs = get_model_identifiers_from_yaml(config.model_family)
         
        self.promptfile = "./dataset/prompt.json"
        with open(self.promptfile, "r") as f:
            self.prompt = json.load(f)
        if config.forget_loss == "idk":
            self.split1, self.split2 = "idk", "retain"
            self.idk = self.prompt['idk']
        elif config.forget_loss == "icd":
            self.split1, self.split2 = "forget", "retain"
            self.forget_data = []
            icd = self.prompt['icd']
            for line in self.forget_personal_data:
                rand_pos = torch.randint(0, len(icd), (1,)).item()
                question = icd[rand_pos].strip(" ").capitalize()
                answer = line['caption'].capitalize()
                for _ in range(10):
                    qa.update(image_path=line['image_path'])
                    qa.update(question=question)
                    qa.update(answer=answer)
                    self.forget_data.append(qa)
        else:
            self.split1, self.split2 = "forget", "retain"
            
        print(
            f"There are {len(self.forget_data)} QA pairs of forget dataset!\n",
            f"There are {len(self.retain_data)} QA pairs of retain dataset!\n",
        )
        
    
    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = [] # (forget, retain)
        for data_type in [self.split1, self.split2]:
            data = self.retain_data if data_type == "retain" else self.forget_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)

            image_path = data[idx]['image_path']
            question = data[idx]['question'].capitalize()
            answer = data[idx]['answer'].capitalize()
            
            if data_type == "idk":
                idk = self.idk
                rand_pos = torch.randint(0, len(idk), (1,)).item()
                answer = idk[rand_pos].strip(" ").capitalize()
            
            if "llava" in self.config.model_family:
                image_tensor = self.image_processor.preprocess(Image.open(image_path), return_tensors='pt')['pixel_values']
                system_message = self.model_configs['system_tag']
                roles = [self.model_configs['question_start_tag'], self.model_configs['answer_tag']]
                conversation = system_message + roles[0] + "<image>\n" + question + roles[1] + answer
                text_input = self.tokenizer(conversation, max_length=self.max_length, truncation=True, return_tensors="pt")
                labels = preprocess_v1(self.tokenizer, text_input['input_ids'], conversation, roles)
                rets.append({**text_input, "labels": labels, "pixel_values": image_tensor})

            elif "llama-3.2" in self.config.model_family.lower():
            
                sources = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": answer}
                    ]},
                ]
                roles = [self.model_configs['question_start_tag'], self.model_configs['answer_tag']]
                input_text = self.processor.apply_chat_template(sources)
                inputs = self.processor(Image.open(image_path), input_text, return_tensors="pt")
                image_tensor = inputs['pixel_values']
                labels = preprocess_v1(self.tokenizer, inputs['input_ids'], input_text, roles)

                input_ids = inputs['input_ids']
                attention_mask = inputs["attention_mask"]
                pixel_values = inputs['pixel_values']
                aspect_ratio_ids = inputs['aspect_ratio_ids']
                aspect_ratio_mask = inputs['aspect_ratio_mask']
                cross_attention_mask = inputs['cross_attention_mask']

                rets.append(dict(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    aspect_ratio_mask=aspect_ratio_mask,
                    aspect_ratio_ids=aspect_ratio_ids,
                    cross_attention_mask=cross_attention_mask,
                    attention_mask=attention_mask,
                    labels=labels,
                ))
            
        return rets
    
 
    
def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks



@dataclass
class custom_data_collator_perturbed(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        max_input_ids_shape = [max(tensor.size(dim) for tensor in input_ids) for dim in range(len(input_ids[0].size()))]
        max_label_shape = [max(tensor.size(dim) for tensor in labels) for dim in range(len(labels[0].size()))]

        pad_input_ids_list, pad_label_list = [], [] 
        for tensor in input_ids:
            padding_width = max_input_ids_shape[1] - tensor.size(1)
            padded_tensor = F.pad(tensor, (0, padding_width), 'constant', self.tokenizer.pad_token_id)
            pad_input_ids_list.append(padded_tensor)

        for tensor in labels:
            padding_width = max_label_shape[1] - tensor.size(1)
            padded_tensor = F.pad(tensor, (0, padding_width), 'constant', -100)
            pad_label_list.append(padded_tensor)
        
        input_ids = torch.stack(pad_input_ids_list)
        labels = torch.stack(pad_label_list)
        
        input_ids = input_ids[:, :, :self.tokenizer.model_max_length]
        labels = labels[:, :, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        
        for key in ['pixel_values', 'aspect_ratio_ids', 'aspect_ratio_mask']:
            if key in instances[0]:
                values = [instance[key].squeeze(1) for instance in instances]
                if all(x is not None and x.shape == values[0].shape for x in values):
                    batch[key] = torch.stack(values)
                else:
                    batch[key] = values
                
                if key == 'pixel_values' and len(values[0].shape) > 4:
                    batch[key] = batch[key].squeeze(1).unsqueeze(0)
                batch[key] = batch[key].squeeze(1)
        
        if "cross_attention_mask" in instances[0]:
            cross_attention_mask_list = [instance["cross_attention_mask"] for instance in instances]
            cross_attention_mask = pad_sequence(
                    cross_attention_mask_list, padding_side='right', padding_value=0
                )
            
            batch['cross_attention_mask'] = cross_attention_mask
          
        if 'qformer_input_ids' in instances[0]:
            qformer_input_ids = [instance['qformer_input_ids'] for instance in instances]
            if all(x is not None and x.shape == qformer_input_ids[0].shape for x in qformer_input_ids):
                batch['qformer_input_ids'] = torch.stack(qformer_input_ids)
            else:
                batch['qformer_input_ids'] = qformer_input_ids
                
            qformer_attention_mask = [instance['qformer_attention_mask'] for instance in instances]
            if all(x is not None and x.shape == qformer_attention_mask[0].shape for x in qformer_attention_mask):
                batch['qformer_attention_mask'] = torch.stack(qformer_attention_mask)
            else:
                batch['qformer_attention_mask'] = qformer_attention_mask
                
        return batch

@dataclass
class custom_data_collator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        
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
        for key in ['pixel_values', 'aspect_ratio_ids', 'aspect_ratio_mask']:
            if key in instances[0]:
                values = [instance[key].squeeze(1) for instance in instances]
                if all(x is not None and x.shape == values[0].shape for x in values):
                    batch[key] = torch.stack(values)
                else:
                    batch[key] = values
                
                if key == 'pixel_values' and len(values[0].shape) > 4:
                    batch[key] = batch[key].squeeze(1).unsqueeze(0)
                else:
                    batch[key] = batch[key].squeeze(1)

        if "cross_attention_mask" in instances[0]:
            cross_attention_mask_list = [instance["cross_attention_mask"][0] for instance in instances]
            cross_attention_mask = pad_sequence(
                    cross_attention_mask_list, padding_side='right', padding_value=0
                )
            
            batch['cross_attention_mask'] = cross_attention_mask
                
        if 'qformer_input_ids' in instances[0]:
            qformer_input_ids = [instance['qformer_input_ids'] for instance in instances]
            if all(x is not None and x.shape == qformer_input_ids[0].shape for x in qformer_input_ids):
                batch['qformer_input_ids'] = torch.stack(qformer_input_ids)
            else:
                batch['qformer_input_ids'] = qformer_input_ids
                
            qformer_attention_mask = [instance['qformer_attention_mask'] for instance in instances]
            if all(x is not None and x.shape == qformer_attention_mask[0].shape for x in qformer_attention_mask):
                batch['qformer_attention_mask'] = torch.stack(qformer_attention_mask)
            else:
                batch['qformer_attention_mask'] = qformer_attention_mask
        
        if 'category' in instances[0]:
            categories = [instance['category'][0] for instance in instances]
            batch['category'] = categories
        
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

            if "cross_attention_mask" in samples[0]:
                cross_attention_mask_list = [instance["cross_attention_mask"][0] for instance in samples]
                cross_attention_mask = pad_sequence(
                        cross_attention_mask_list, padding_side='right', padding_value=0, max_len=input_ids.shape[-1]
                    )
                batch['cross_attention_mask'] = cross_attention_mask
                
            for key in ['pixel_values', 'aspect_ratio_ids', 'aspect_ratio_mask']:
                if key in samples[0]:
                    values = [instance[key].squeeze(1) for instance in samples]
                    if all(x is not None and x.shape == values[0].shape for x in values):
                        batch[key] = torch.stack(values)
                    else:
                        batch[key] = values
                    
                    
                    batch[key] = batch[key].squeeze(1)

                   

            rets.append(batch)
                
        return rets


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()
    
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(-1)
    return loss
