CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 accelerate launch \
    --config_file config/accelerate_config.yaml \
    ./forget.py --config-name forget.yaml \