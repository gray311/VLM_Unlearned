# VFUBench: Benchmarking Vision Language Model Unlearning via Fictious Datasets

## Introduction

We introduce **VFUBench**, a new benchmark designed to robustly evaluate VLM unlearning, especially under the **Right to be Forgotten** setting. This benchmark is built on our newly developed Fictitious Entity Visual Question Answering (VQA) dataset, which includes 400 virtual entities, each represented through 20 VQAs focusing on image-related attributes and background knowledge. 

## Fictitious Datasets

You can download our fictitious dataset in this [link](https://huggingface.co/datasets/gray311/VFUBench). Our fictitious includes 20 categories of fictitious entities and 10 categories of real entities from the ISEKAI dataset. Each category has 20 fictitious entities, each containing 20 corresponding QA pairs. 
```
    Fictitious category: ['cactus boxer', 'cactus hedgehog', 'fire snowman', 'flying jellyfish', 'goldfish airship', 'horned elephant', 'Ice cream microphone', 'magma snake', 'muscle tiger', 'mushroom house', 'octopus vacuum cleaner', 'panda with wings', 'pineapple house', 'rhino off-road vehicle', 'robofish', 'rock sheep', 'suspended rock', 'transparent deer', 'turtle castle', 'zebra striped rabbit']
    Real category: ['cactus', 'hedgehog', 'Ice cream', 'mushroom', 'pineapple', 'rock', 'sheep', 'snake', 'tiger', 'zebra']
```
