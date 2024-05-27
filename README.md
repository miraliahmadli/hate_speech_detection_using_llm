# EPFL Deep Learning 2024 Project - Robust Hate Speech Detection for LLMs

## Project Description

- For this project, we use SOTA techniques to improve LLMs hate speech detection accuracy.
- First, we fine-tune our baseline LLM for the task. Then, we use KTO to align our model according to human preferences.
- Finally, we use RAG to try to improve our model's performance even further. 

## Codebase File Structure

```txt
.
├── checkpoints
│   ├── checkpoint_name
│   │   ├── config.json
│   │   |── model.safetensor
│   │   └── ...
├── data
│   │   ├── dataset_name
│   │   │   └── ...
│   │   └── ...
├── dataloaders
│       ├── common_utils.py
│       ├── dpo_dataset.py
│       ├── kto_dataset.py
│       └── sft_dataset.py
├── documents (For RAG only)
├── models
│       ├── model_base.py
│       |── model_sft.py
│       └── ...
├── RAG
│       ├── document_scraper.py
│       └── retriever.py
├── evaluate_model.py
├── train_base.py
├── train_kto.py
├── train_sft.py
├── train_sft_kto.py
├── requirements.txt
└── README.md
```

## Setup

### Setup via Conda Virtual Environment

```bash
# Replace <my-env> with the name of your environment.
conda create --name <my-env> python=3.10.11
conda activate <my-env>

# Install dependencies from a `requirements.txt`
pip install -r requirements.txt
```


## Training Models
- `train_sft.py` contains codebase for finetuning GPT2 model. Modify pathes and configs in the file for your wishes and run `python train_sft.py` to start training.
- `train_kto.py` contains codebase for aligning model using KTO. Modify pathes and configs in the file for your wishes and run `python train_kto.py` to start training.
  - __NOTE__: KTOTrainer has known OOM issue and hasn't been fixed as of yet.
- `train_sft_kto.py` contains codebase for finetuning GPT2 model that is aligned using KTO. Modify model name to your KTO checkpoint path, update pathes and configs in the file for your wishes and run `python train_sft_kto.py` to start training.
  - If you used LoRA for KTO, consider merging using `merge_and_push` method provided in `models/model_kto.py` file to merge LoRA matrices to model. 
- `train_base.py` can be used to fine tune models on variety of available Language Models. We used it for fine-tuning some BERT based models.

## Evaluate Models
- `evaluate_model.py`: update the path to model checkpoint, specify datasets you want to evaluate and run the python script.
