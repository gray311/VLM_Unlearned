CUDA_VISIBLE_DEVICES=1 DS_SKIP_CUDA_CHECK=1 accelerate launch \
    --config_file config/accelerate_config.yaml \
    --main_process_port 8888 \
    ./forget.py --config-name forget.yaml \