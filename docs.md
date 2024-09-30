## Install

```
conda create -n vlm python=3.10
conda activate vlm
git clone -b face https://github.com/gray311/VLM_Unlearned.git
cd VLM_Unlearned

pip install -r requirements.txt
```

## Prepare dataset

```
mkdir dataset
cd dataset
git clone https://huggingface.co/datasets/gray311/FIUBench

cd FIUBench
unzip SFHQ.zip
mv * ./../

cd ..
rm -rf FIUBench
```

## Inference

```
### model 1: retain

python inference.py \
    --eval_file ./dataset/full.json \
    --split forget5 \
    --loss_type full \
    --model_path gray311/vlm_unlearning_ft_llava_phi_3_mini_retain \
    --model_name llava-phi 

### model 2: unlearned-GA

python inference.py \
    --eval_file ./dataset/full.json \
    --split forget5 \
    --loss_type ga \
    --model_path gray311/vlm_unlearning_ft_llava_phi_3_mini \
    --model_name llava-phi 

### model 3: unlearned-GD

python inference.py \
    --eval_file ./dataset/full.json \
    --split forget5 \
    --loss_type gd \
    --model_path gray311/vlm_unlearning_ft_llava_phi_3_mini \
    --model_name llava-phi 

### model 4: unlearned-KL

python inference.py \
    --eval_file ./dataset/full.json \
    --split forget5 \
    --loss_type kl \
    --model_path gray311/vlm_unlearning_ft_llava_phi_3_mini \
    --model_name llava-phi 

### model 5: unlearned-PO

python inference.py \
    --eval_file ./dataset/full.json \
    --split forget5 \
    --loss_type idk \
    --model_path gray311/vlm_unlearning_ft_llava_phi_3_mini \
    --model_name llava-phi 


### model 5: unlearned-ICD

python inference.py \
    --eval_file ./dataset/full.json \
    --split forget5 \
    --loss_type icd \
    --model_path gray311/vlm_unlearning_ft_llava_phi_3_mini \
    --model_name llava-phi 
```