DS_SKIP_CUDA_CHECK=1 CUDA_VISIBLE_DEVICES=3 accelerate launch \
    --config_file config/accelerate_config.yaml \
     --main_process_port 2216 \
    ./forget.py --config-name forget_lora.yaml \