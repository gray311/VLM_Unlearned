import os 
import sys
import time 
import json
import math
import copy
import gc
from tqdm import tqdm
import hydra
import datasets
import logging
import requests
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model
import transformers
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_scheduler,
    SchedulerType
)
from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    set_seed, 
    LlavaForConditionalGeneration, 
    AutoProcessor,
    CLIPImageProcessor
)
import deepspeed
from transformers.integrations.deepspeed import (
    deepspeed_init, 
    deepspeed_load_checkpoint, 
    is_deepspeed_available
)
from utils import (
    get_model_identifiers_from_yaml, 
    get_cast_dtype, 
    parse_pred_ans,
    save_lora_weights
)

from data_module import MMForgetDatasetQA, custom_data_collator, custom_data_collator_forget
from data_loader import CustomTrainer



logger = get_logger(__name__)


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_tower', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
  
def get_grouped_params(model):
        def apply_decay(x):
            return "bias" not in x

        return [
            {
                "params": [
                    p for n, p in model.named_parameters() if p.requires_grad and apply_decay(n)
                ],
                "weight_decay": 0.01
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if p.requires_grad and not apply_decay(n)
                ],
                "weight_decay": 0.0
            }
        ]
          
def get_optimizer(config, model):
    return torch.optim.AdamW(get_grouped_params(model), lr=config.lr)
    

def e_prepare_deepspeed(model, accelerator):
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)
    
    if model is not None:
        if hasattr(model, "config"):
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                    }
                )

    # If ZeRO-3 is used, we shard both the active and reference model.
    # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
    if config_kwargs["zero_optimization"]["stage"] != 3:
        config_kwargs["zero_optimization"]["stage"] = 0
    config_kwargs["optimizer"] = {"type": None}

    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    #set the gradients to false for every parameter
    for param in model.parameters():
        param.requires_grad = False
    
    return model
    

@hydra.main(version_base=None, config_path="config", config_name="forget")
def main(cfg):
    set_seed(cfg.seed)
    
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    accelerator_log_kwargs = {}
    accelerator_log_kwargs["log_with"] = cfg.report_to
    accelerator_log_kwargs["project_dir"] = cfg.save_dir
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        **accelerator_log_kwargs)

    if accelerator.is_main_process:
        if cfg.save_dir is not None:
            os.makedirs(cfg.save_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(cfg.save_dir, "log.txt"))
        ] if accelerator.is_main_process else [])
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        
        
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"] 
    # save the cfg file
    #if master process
    if accelerator.is_main_process:
        with open(f'{cfg.save_dir}/cfg.yaml', 'w') as f:
            OmegaConf.save(cfg, f)
            
    oracle_model = None
    if "llava" in cfg.model_path:
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        model = LlavaForConditionalGeneration.from_pretrained(cfg.model_path, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
        if cfg.forget_loss == "KL":
            oracle_model = LlavaForConditionalGeneration.from_pretrained(cfg.model_path, attn_implementation="flash_attention_2", torch_dtype=torch.float16)

    if cfg.LoRA.r != 0:
        target_modules=r'.*language_model.*\.(up_proj|k_proj|linear_2|down_proj|v_proj|q_proj|o_proj|gate_proj|linear_1)'
        config = LoraConfig(
            r=cfg.LoRA.r, 
            lora_alpha=cfg.LoRA.alpha, 
            target_modules=target_modules, 
            lora_dropout=cfg.LoRA.dropout,
            bias="none", 
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)

        for n, p in model.named_parameters():
            if cfg.tune_vision_tower and "vision_tower" in n:
                p.requires_grad = True
            if cfg.tune_mm_projector and "projector" in n:
                p.requires_grad = True
                
    
    max_length = 512
    torch_format_dataset = MMForgetDatasetQA(config=cfg, tokenizer=tokenizer, image_processor=image_processor, max_length=max_length)

    
    batch_size, workers = cfg.batch_size, cfg.workers
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    
    torch_format_dataloader = DataLoader(
        torch_format_dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True,
        collate_fn=custom_data_collator_forget(tokenizer=tokenizer),
    )
    

    if cfg.LoRA.r == 0:
        for n, p in model.named_parameters():
            if not cfg.tune_vision_tower and "vision_tower" in n:
                p.requires_grad = False
            if not cfg.tune_mm_projector and "projector" in n:
                p.requires_grad = False
            if not cfg.tune_language_model and "language_model" in n:
                p.requires_grad = False
            
    
    optimizer = get_optimizer(cfg, model)
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(torch_format_dataloader) / (gradient_accumulation_steps * accelerator.num_processes))
    max_train_steps = cfg.num_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=cfg.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=round(cfg.warmup_ratio * max_train_steps),
        num_training_steps=max_train_steps,
    )

    
    if accelerator.is_main_process:
        print_trainable_parameters(model)
        
    model, optimizer, torch_format_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, torch_format_dataloader, lr_scheduler)
    if cfg.forget_loss == "KL":
        oracle_model = e_prepare_deepspeed(oracle_model, accelerator)
    
    accelerator.init_trackers(project_name="vlm_unlearned")
    total_batch_size = batch_size * accelerator.num_processes * gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(torch_format_dataset)}")
    logger.info(f"  Num Epochs = {cfg.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Total warmup steps = {int(cfg.warmup_ratio * max_train_steps)}")
    

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(int(max_train_steps)), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    
    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint is not None or cfg.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {cfg.resume_from_checkpoint}")
            accelerator.load_state(cfg.resume_from_checkpoint)
            path = os.path.basename(cfg.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * gradient_accumulation_steps
            starting_epoch = resume_step // len(torch_format_dataloader)
            resume_step -= starting_epoch * len(torch_format_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch
    
    
    for epoch in range(starting_epoch, cfg.num_epochs):
        model.train()
        total_loss = 0
        losses = []
        kl_losses = []
        cast_dtype  = get_cast_dtype(accelerator.mixed_precision)

        for step, batch in enumerate(torch_format_dataloader):
            # We need to skip steps until we reach the resumed step
            if cfg.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    if step % gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue
            
            forget_inputs, retain_inputs = batch

            with accelerator.accumulate(model):
                if cfg.forget_loss == "grad_ascent":
                    outputs = model(**forget_inputs)
                    loss = outputs.loss
                    loss = loss * -1
                
                #preference optimization
                elif cfg.forget_loss == "idk":
                    input_ids = torch.cat((forget_inputs['input_ids'], retain_inputs['input_ids']), dim=0)
                    labels = torch.cat((forget_inputs['labels'], retain_inputs['labels']), dim=0)
                    attention_mask = torch.cat((forget_inputs['attention_mask'], retain_inputs['attention_mask']), dim=0)
                    pixel_values = torch.cat((forget_inputs['pixel_values'], retain_inputs['pixel_values']), dim=0)
                    outputs = model(**{'input_ids': input_ids, "labels": labels, "attention_mask": attention_mask, "pixel_values": pixel_values})
                    loss = outputs.loss
                        
                #minimum KL divergence
                elif cfg.forget_loss == "KL":
                    outputs = model(**forget_inputs)
                    loss = outputs.loss
                    loss = loss * -1

                    with torch.no_grad():
                        retain_outputs = oracle_model(**retain_inputs)
                    retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
                    retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

                    current_outputs = model(**retain_inputs)
                    current_probs = F.log_softmax(current_outputs.logits, dim=-1)
                    current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])
                    kl_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)
                    kl_losses.append(kl_loss.detach().float())

                    loss = loss + kl_loss
                    
                progress_bar.set_description(
                    f"Epoch {epoch} - Step {step} - LR: {optimizer.param_groups[0]['lr']:.2e} - loss: {loss:.4f}")

                total_loss += loss.detach().float()
                losses.append(loss.detach().float())
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), cfg.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                accumulate_loss = torch.tensor(losses)
                accumulate_loss = accumulate_loss[~torch.isnan(accumulate_loss)]
                
                if len(kl_losses) > 0:
                    accumulate_kl_loss = torch.tensor(kl_losses)
                    accumulate_kl_loss = accumulate_kl_loss[~torch.isnan(accumulate_kl_loss)]
                    losses,  kl_losses = [], []
                    accelerator.log(
                        {
                            "loss": torch.mean(accumulate_loss).item(),
                            "kl_loss": torch.mean(accumulate_kl_loss).item(),
                            "step": completed_steps,
                            "learning_rate": optimizer.param_groups[0]['lr'],
                        },
                        step=completed_steps,
                    )
                else:
                    accelerator.log(
                        {
                            "loss": torch.mean(accumulate_loss).item(),
                            "step": completed_steps,
                            "learning_rate": optimizer.param_groups[0]['lr'],
                        },
                        step=completed_steps,
                    )
                
                if cfg.save_steps > 0 and completed_steps % cfg.save_steps == 0:
                    accelerator.wait_for_everyone()
                    output_dir = f"step_{completed_steps}"
                    if cfg.save_dir is not None:
                        output_dir = os.path.join(cfg.save_dir, output_dir)
                    if accelerator.is_main_process:
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        
                        unwrapped_model = accelerator.unwrap_model(model)
                        
                        if cfg.LoRA.r != 0:
                            save_lora_weights(unwrapped_model, output_dir)
                        else:
                            unwrapped_model.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)
                            
                        gc.collect()
                        torch.cuda.empty_cache()
                    

                if completed_steps >= max_train_steps:
                    break

    accelerator.end_training()
    output_dir = cfg.save_dir
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        try:
            os.makedirs(output_dir)
        except OSError:
            pass
        
        unwrapped_model = accelerator.unwrap_model(model)
        #save the model
        if cfg.LoRA.r != 0:
            save_lora_weights(unwrapped_model, output_dir)
        else:
            unwrapped_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        
if __name__ == "__main__":
    main()
