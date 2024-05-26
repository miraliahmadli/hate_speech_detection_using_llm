import numpy as np
from trl import SFTConfig
from peft import LoraConfig
import os
os.environ['WANDB_DISABLED'] = 'true'

from datasets import concatenate_datasets
# from models.model_sft import sft_pipeline
from models.model_gpt import sft_pipeline
from dataloaders.common_utils import *


class Config:
    beta = 0.1 # the beta parameter for DPO loss
    desirable_weight = 1.0
    undesirable_weight = 1.0

    # training parameters
    model_name_or_path = "../checkpoint/sft2_kto"
    learning_rate = 5e-4
    lr_scheduler_type = "cosine"
    optimizer_type = "paged_adamw_32bit"
    batch_size = 32
    lora_alpha = 16
    lora_dropout = 0.05
    lora_r = 8

    max_prompt_length = 256
    max_length = 256
    num_train_epochs = 2
    logging_steps = 1000
    save_steps = 2000
    eval_steps = 2000
    output_dir = "./results/sft2_kto/"
    log_freq = 1

    # instrumentation
    report_to=None
    seed = 0
 
 
def return_prompt_and_responses(samples):
    return {
        "prompt": samples["prompt"],
        "completion": samples["completion"],
        "label": samples["label"],
    }

def filter_none_text(example, key='prompt'):
    return example[key] is not None

   
    
    
if __name__ == "__main__":
    MODEL_NAME = 'checkpoints/final_kto_merged_checkpoint' # kto aligned
    MAX_LEN = 256
    TRAIN_BATCH_SIZE = 32
    EVAL_BATCH_SIZE = 32
    EPOCHS = 3
    LEARNING_RATE = 1e-5
    NUM_LABELS = 2
    OUTPUT_DIR = './checkpoints/sft_kto'
    CKPT = './checkpoints/finetune_model_kto'

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        evaluation_strategy = "steps",
        num_train_epochs = EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        logging_steps=1500,
        learning_rate=LEARNING_RATE,
        save_steps=1000,
        weight_decay=0.01,
        report_to="tensorboard",
        logging_dir="./tensorboard/sft_gpt2_kto",
        model_init_kwargs=None)

    peft_config = LoraConfig(
        r=Config.lora_r,
        lora_alpha=Config.lora_alpha,
        lora_dropout=Config.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    
    tg_train, tg_val = load_and_process_toxigen()
    tw_train, tw_val = load_and_process_twitter_data()
    berkeley_train, berkeley_val = load_and_process_berkeley_data()
    gender_train, gender_val = load_and_process_gender_hate_speech_data()
    cad_train, cad_val = load_and_process_cad()
    train_datasets = [tg_train, tw_train, berkeley_train, gender_train, cad_train]
    val_datasets = [tg_val, tw_val, berkeley_val, gender_val, cad_val]

    concat_train = concatenate_datasets(train_datasets)
    concat_val = concatenate_datasets(val_datasets)


    sft_pipeline(MODEL_NAME, 
        training_args,
        concat_train,
        concat_val,
        ckpt_path="./checkpoints/final_sft_checkpoint_kto")
