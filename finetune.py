import os 
import sys
import time 
import json
import math
from tqdm import tqdm
import hydra
import datasets
import logging
import requests
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import transformers
from transformers import (get_constant_schedule_with_warmup,
                          get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup,
                          get_scheduler,
                          SchedulerType)

from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    set_seed, 
    LlavaForConditionalGeneration, 
    AutoProcessor,
    CLIPImageProcessor
)
from peft import LoraConfig, get_peft_model

from utils import get_model_identifiers_from_yaml, get_cast_dtype
from data_module import MMDatasetQA, custom_data_collator
from data_loader import CustomTrainer

logger = get_logger(__name__)


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
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


@hydra.main(version_base=None, config_path="config", config_name="finetune")
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
            

    if "llava" in cfg.model_id:
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        model = LlavaForConditionalGeneration.from_pretrained(cfg.model_id, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
        
        if cfg.LoRA.r != 0:
            config = LoraConfig(
                r=cfg.LoRA.r, 
                lora_alpha=cfg.LoRA.alpha, 
                target_modules=find_all_linear_names(model), 
                lora_dropout=cfg.LoRA.dropout,
                bias="none", 
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, config)
       
        max_length = 512
        torch_format_dataset = MMDatasetQA(config=cfg, tokenizer=tokenizer, image_processor=image_processor, max_length=max_length)
        
        batch_size, workers = cfg.batch_size, cfg.workers
        gradient_accumulation_steps = cfg.gradient_accumulation_steps
        
        torch_format_dataloader = DataLoader(
            torch_format_dataset,
            batch_size=batch_size,
            num_workers=workers,
            shuffle=True,
            collate_fn=custom_data_collator(tokenizer=tokenizer),
        )
        
        for n, p in model.named_parameters():
            if not cfg.tune_vision_tower and "vision_tower" in n:
                p.requires_grad = False
                
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
        
        optimizer = torch.optim.AdamW(get_grouped_params(model), lr=cfg.lr)
        
        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(torch_format_dataloader) / gradient_accumulation_steps)
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
            cast_dtype  = get_cast_dtype(accelerator.mixed_precision)

            for step, batch in enumerate(torch_format_dataloader):
                # We need to skip steps until we reach the resumed step
                if cfg.resume_from_checkpoint and epoch == starting_epoch:
                    if resume_step is not None and step < resume_step:
                        if step % gradient_accumulation_steps == 0:
                            progress_bar.update(1)
                            completed_steps += 1
                        continue

                with accelerator.accumulate(model):
                    loss = model(**batch).loss

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
                    # filter out nan
                    accumulate_loss = accumulate_loss[~torch.isnan(accumulate_loss)]
                    losses = []
                    accelerator.log(
                        {
                            "loss": torch.mean(accumulate_loss).item(),
                            "step": completed_steps,
                            "learning_rate": optimizer.param_groups[0]['lr'],
                        },
                        step=completed_steps,
                    )
        
                    if cfg.save_steps > 0 and completed_steps % cfg.save_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if cfg.save_dir is not None:
                            output_dir = os.path.join(cfg.save_dir, output_dir)
                        if accelerator.is_main_process:
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            unwrapped_model = accelerator.unwrap_model(model)
                            
                            #save the model
                            if cfg.LoRA.r != 0:
                                unwrapped_model = unwrapped_model.merge_and_unload()
                                
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
                unwrapped_model = unwrapped_model.merge_and_unload()
                
            unwrapped_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        

        # url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
        # image = Image.open(requests.get(url, stream=True).raw)
        # prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nWhat is shown in this image? ASSISTANT:"
        # image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        # image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(model.device)
        
        # print(image_tensor.shape)
        # text_input = tokenizer(prompt)
        # print(text_input)
        
        
        # text_input = {k: v.to(model.device) for k, v in text_input.items()}
        
        # print(text_input['input_ids'])
        
        # inputs = {**text_input, "pixel_values": image_tensor}
        # output = model.generate(**inputs, max_new_tokens=1024)

        # print(tokenizer.decode(output[0], skip_special_tokens=True))
            
        
if __name__ == "__main__":
    main()