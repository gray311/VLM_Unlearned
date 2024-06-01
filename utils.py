import yaml
import copy
import numpy as np
import torch

def get_model_identifiers_from_yaml(model_family):
    #path is model_configs.yaml
    '''
    models:
        llama2-7b:
            hf_key: "NousResearch/Llama-2-7b-chat-hf"
            question_start_tag: "[INST] "
            question_end_tag: " [/INST] "
            answer_tag: ""
            start_of_sequence_token: "<s>"
    '''
    model_configs  = {}
    with open("config/model_config.yaml", "r") as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    return model_configs[model_family]


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype

def parse_pred_ans(pred_ans):
    pred_label = None
    if pred_ans in ["yes", "no"]:
        pred_label = pred_ans
    else:
        prefix_pred_ans = pred_ans[:4]

        if "yes" in prefix_pred_ans:
            pred_label = "yes"
        elif "no" in prefix_pred_ans:
            pred_label = "no"
        else:
            pred_label = "other"

    return pred_label 


def filter_state_dict_to_trainable(model, state_dict):
    for (name, p,) in model.named_parameters():  # won't work for fsdp + use_orig_params=False
        if "fsdp" in name:
            continue
        if "embed" in name or isinstance(p, torch.nn.Embedding):
            continue
        if not p.requires_grad:
            name = name.replace("._checkpoint_wrapped_module", "")
            if name in state_dict:
                del state_dict[name]
            else:
                print(f"WARNING: filtering but {name} not in state_dict")


    to_delete = [
        n
        for n in state_dict.keys()
        or ("vision_tower" in n)
        or ("embed_tokens" in n)
    ]
    
    for name in to_delete:
        del state_dict[name]
        
    for k, v in state_dict.items():
        print(k, v.shape)
    
    return state_dict

def save_lora_weights(model, output_dir):
    """
    Save training checkpoint with model, optimizer, and lr_scheduler state.
    """
 
    trained_params = {name: param.to(torch.float16).cpu() for name, param in model.named_parameters() if param.requires_grad}

    print(f"Saving checkpoint to {output_dir}/checkpoint.pt")
    torch.save(trained_params, f"{output_dir}/checkpoint.pt")
