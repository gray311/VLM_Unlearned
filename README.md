# VFUBench: Benchmarking Vision Language Model Unlearning via Fictious Datasets

## Introduction

We introduce **VFUBench**, a new benchmark designed to robustly evaluate VLM unlearning, especially under the **Right to be Forgotten** setting. This benchmark is built on our newly developed Fictitious Entity Visual Question Answering (VQA) dataset, which includes 400 virtual entities, each represented through 20 VQAs focusing on image-related attributes and background knowledge. 

## Fictitious Datasets

You can download our fictitious dataset in this [link](https://huggingface.co/datasets/gray311/VFUBench). Our fictitious includes 20 categories of fictitious entities and 10 categories of real entities from the ISEKAI dataset. Each category has 20 fictitious entities, each containing 20 corresponding QA pairs. 
```
    Fictitious category: ['cactus boxer', 'cactus hedgehog', 'fire snowman', 'flying jellyfish', 'goldfish airship', 'horned elephant', 'Ice cream microphone', 'magma snake', 'muscle tiger', 'mushroom house', 'octopus vacuum cleaner', 'panda with wings', 'pineapple house', 'rhino off-road vehicle', 'robofish', 'rock sheep', 'suspended rock', 'transparent deer', 'turtle castle', 'zebra striped rabbit']
    Real category: ['cactus', 'hedgehog', 'Ice cream', 'mushroom', 'pineapple', 'rock', 'sheep', 'snake', 'tiger', 'zebra']
```

1. For learning (stage 1):

| Full | Retain |
|------|--------|
| 7971 | 6806   |

2. For unlearning (stage 2):

| Overall Split |            |             |
|---------------|------------|-------------|
| Forget set    | Retain set | Reality set |
| 805           | 7166       | 200         |

| Fine-grained Split |            |             |
|--------------------|------------|-------------|
| Forget set         | Retain set | Reality set |
| 723                | 723        | 100         |

## Unlearning Pipeline

### Install

1. Clone this repository and navigate to VLM_Unlearned folder

```
git clone https://github.com/gray311/VLM_Unlearned.git
cd VLM_Unlearned
```

2. Install Package
```
conda create -n unlearned python=3.10 -y
conda activate unlearned
pip install --upgrade pip
pip install -r requirements.txt
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### Data Preparation

1. Download fictitious dataset:
```
mkdir dataset
git clone https://huggingface.co/datasets/gray311/VFUBench/
```
### Learning

1. Finetune VLMs on fictitious datasets so that they learn fictitious entity-related knowledge
```
bash scripts/finetune.bash

# you can modify config/accelerate.yaml and finetune.yaml according to your expected settings.
```







