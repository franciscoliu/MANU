<h2 align="center"> <a href="https://arxiv.org/abs/2502.15910">Modality-Aware Neuron Pruning for Unlearning in Multimodal Large Language Models (MANU) </a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>

## Abstract 
Generative models such as Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) trained on massive datasets can lead them to memorize and inadvertently reveal sensitive information, raising ethical and privacy concerns. While some
prior works have explored this issue in the context of LLMs, it presents a unique challenge for MLLMs due to the entangled nature of knowledge across modalities, making comprehensive unlearning more difficult. To address this challenge, we propose Modality Aware Neuron Unlearning (MANU), a novel unlearning framework for MLLMs designed to selectively clip neurons based on their relative importance to the targeted forget data, curated for different modalities. Specifically, MANU consists of two stages: important neuron selection and selective pruning. The first stage identifies and collects the most influential neurons across modalities relative to the targeted forget knowledge, while the second stage is dedicated to pruning those selected neurons. MANU effectively isolates and removes the neurons that contribute most to the forget data within each modality, while preserving the integrity of retained knowledge. Our experiments conducted across various MLLM architectures illustrate that MANU can achieve a more balanced and comprehensive unlearning in each modality without largely affecting the overall model utility.

## News :mega:
- **[May 15, 2025]** MANU has been accepted by ACL 2025 Main!

## Installation
Firstly, you need to create a environment with `python=3.10`:
```
conda create -n mllm_unlearn_prune python=3.10
conda activate mllm_unlearn_prune
pip install -r requirements.txt
```
## Data Processing
First of all, you need to obtain MLLMU-Bench from its offical [site](https://github.com/franciscoliu/MLLMU-Bench) and obtain a separate prune_data. The overall structure should look like following:
```
data/
├── MLLMU (follow the instructions from official [MLLMU-Bench](https://github.com/franciscoliu/MLLMU-Bench))
│   ├── Full_Set/
│   │   └── train-00000-of-00001.parquet
│   └── Retain_Set/
│       └── train-00000-of-00001.parquet
├── prune_data/
│   ├── forget_5/
│   │   ├── forget_ids.json
│   │   ├── qa_files/
│   │   └── train-00000-of-00001.parquet
│   ├── forget_10/
│   │   ├── forget_ids.json
│   │   ├── qa_files/
│   │   └── train-00000-of-00001.parquet
│   ├── forget_15/
│   │   ├── forget_ids.json
│   │   ├── qa_files/
│   │   └── train-00000-of-00001.parquet
│   ├── retain_5/
│   │   ├── forget_ids.json
│   │   ├── qa_files/
│   │   └── train-00000-of-00001.parquet
│   ├── retain_10/
│   │   ├── forget_ids.json
│   │   ├── qa_files/
│   │   └── train-00000-of-00001.parquet
│   └── retain_15/
│       ├── forget_ids.json
│       ├── qa_files/
│       └── train-00000-of-00001.parquet
```
You can download `prune_data` [here](https://drive.google.com/drive/folders/10tnKyG_1ij0MlvuxSa3NJ26x54H4GB9C?usp=sharing) from MLLMU-Bench for your convenience. There's nothing unusual about prune_data; it's simply been reorganized to better support the upcoming modality-aware pruning. Please open an issue if you have trouble downloading the data. 

## Pruning
After setting up the data, you may run pruning process by calling the following command:
```
python prune/prune_incremental.py \
	--model_save_name llava-1.5-7b \
	--model_id llava-hf/llava-1.5-7b-hf \
	--vanilla_dir LLaVA_7b_hf_vanilla \
	--data_dir data/prune_data \
  --batch_size 4  \
	--max_length 384 \
	--forget_ratio 15 \
	--prune_percent 10 \
	--save_path [Pruned_Model_Saved_Path]
```
- `vanilla_dir` is the [vanilla model](https://huggingface.co/MLLMMU) directory provided by MLLMU_Bench.
- `data_dir` is the aforementioned post-processed MLLMU_Bench data for pruning.
You may also modify `forget_ratio` and `prune_percent`.

## Evaluation
To run the evaluation of pruned model (or any saved model), you can run the following command:
```
python eval.py \
 --model_id llava-hf/llava-1.5-7b-hf \
 --cache_path [Saved_Pruned_Model_Path] \
 --test_data data/MLLMU-Bench/Test_Set \
 --few_shot_data data/MLLMU-Bench/Full_Set/train-00000-of-00001.parquet \
 --data_split_folder data/MLLMU-Bench \
 --celebrity_data data/MLLMU-Bench/Retain_Set/train-00000-of-00001.parquet \
 --output_file  llava-1.5-7b_percent_10.0_iteration_1_forget_5\
 --output_folder eval_result/llava-1.5-7b_percent_10.0_iteration_1_forget_5 \
 --forget_ratio 5 \
```
- `cache_path` is the saved post-pruned model from the pruning process

## Citing Our Work :star2:
If you find our work helpful or inspiring, please consider citing our work:
```
@article{liu2025modality,
  title={Modality-Aware Neuron Pruning for Unlearning in Multimodal Large Language Models},
  author={Liu, Zheyuan and Dou, Guangyao and Yuan, Xiangchi and Zhang, Chunhui and Tan, Zhaoxuan and Jiang, Meng},
  journal={arXiv preprint arXiv:2502.15910},
  year={2025}
}
```

