from trl import SFTConfig
from peft import LoraConfig
import os
os.environ['WANDB_DISABLED'] = 'true'

from datasets import concatenate_datasets
from models.model_sft import sft_pipeline
from dataloaders.common_utils import *


class Config:
    beta = 0.1 # the beta parameter for DPO loss
    desirable_weight = 1.0
    undesirable_weight = 1.0

    # training parameters
    model_name_or_path = "../checkpoint/sft2"
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
    output_dir = "./results/sft2/"
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
    # train, val = load_and_process_toxigen(kto=True)
    
    # training_args = KTOConfig(
    #     beta=Config.beta,
    #     desirable_weight=Config.desirable_weight,
    #     undesirable_weight=Config.undesirable_weight,
    #     per_device_train_batch_size=Config.batch_size,
    #     num_train_epochs=Config.num_train_epochs,
    #     logging_steps=Config.logging_steps,
    #     save_steps=Config.save_steps,
    #     learning_rate=Config.learning_rate,
    #     evaluation_strategy="steps",
    #     eval_steps=Config.eval_steps,
    #     output_dir=Config.output_dir,
    #     lr_scheduler_type=Config.lr_scheduler_type,
    #     optim=Config.optimizer_type,
    #     bf16=False,
    #     remove_unused_columns=False,
    #     run_name="sft_toxigen",
    #     seed=Config.seed,
    #     max_prompt_length=Config.max_prompt_length,
    #     max_length=Config.max_length,
    #     report_to=Config.report_to
    # )

    MODEL_NAME = 'gpt2'
    MAX_LEN = 256
    TRAIN_BATCH_SIZE = 16
    EVAL_BATCH_SIZE = 16
    EPOCHS = 2
    LEARNING_RATE = 5e-4
    NUM_LABELS = 2
    OUTPUT_DIR = './checkpoints/gpt2_base/'
    CKPT = './checkpoints/finetune_model/gpt2_base/'

    training_args = SFTConfig(
      output_dir=OUTPUT_DIR,
      evaluation_strategy = "steps",
      num_train_epochs = EPOCHS,
      per_device_train_batch_size=TRAIN_BATCH_SIZE,
      per_device_eval_batch_size=EVAL_BATCH_SIZE,
      logging_steps=1000,
      learning_rate=LEARNING_RATE,
      save_steps=2000,
      weight_decay=0.01,
      report_to="tensorboard",
      logging_dir="./tensorboard/sft_gpt2",)

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
    
    MODEL_NAME = 'gpt2' # only accepts models with decoders
    
    # train = train.map(
    #     return_prompt_and_responses,
    #     batched=True,
    #     num_proc=8,
    # ).filter(filter_none_text)

    # val = val.map(
    #     return_prompt_and_responses,
    #     batched=True,
    #     num_proc=8,
    # ).filter(filter_none_text)
    
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
                peft_config=peft_config,
                ckpt_path="./checkpoints/final_sft_checkpoint")
