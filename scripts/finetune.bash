DS_SKIP_CUDA_CHECK=1 accelerate launch \
    --config_file config/accelerate_config.yaml \
    ./finetune.py \