from trl import KTOConfig
from peft import LoraConfig
import os
os.environ['WANDB_DISABLED'] = 'true'


from models.model_kto import kto_pipeline
from dataloaders.common_utils import *


class Config:
    beta = 0.1 # the beta parameter for DPO loss
    desirable_weight = 1.0
    undesirable_weight = 1.0

    # training parameters
    model_name_or_path = "../checkpoint/kto2"
    learning_rate = 5e-4
    lr_scheduler_type = "cosine"
    optimizer_type = "paged_adamw_32bit"
    batch_size = 16
    lora_alpha = 16
    lora_dropout = 0.05
    lora_r = 8

    max_prompt_length = 256
    max_length = 256
    num_train_epochs = 2
    logging_steps = 1000
    save_steps = 2000
    eval_steps = 2000
    output_dir = "./results/kto2/"
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
    train, val = load_and_process_toxigen(kto=True)
    
    training_args = KTOConfig(
        beta=Config.beta,
        desirable_weight=Config.desirable_weight,
        undesirable_weight=Config.undesirable_weight,
        per_device_train_batch_size=Config.batch_size,
        num_train_epochs=Config.num_train_epochs,
        logging_steps=Config.logging_steps,
        save_steps=Config.save_steps,
        learning_rate=Config.learning_rate,
        evaluation_strategy="steps",
        eval_steps=Config.eval_steps,
        output_dir=Config.output_dir,
        lr_scheduler_type=Config.lr_scheduler_type,
        optim=Config.optimizer_type,
        bf16=False,
        remove_unused_columns=False,
        run_name="kto_toxigen",
        seed=Config.seed,
        max_prompt_length=Config.max_prompt_length,
        max_length=Config.max_length,
        report_to=Config.report_to,
        max_grad_norm=1.0,
    )

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
    
    train = train.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=8,
    ).filter(filter_none_text)

    val = val.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=8,
    ).filter(filter_none_text)
    
    
    kto_pipeline(MODEL_NAME, 
                training_args,
                train,
                val,
                peft_config=peft_config,
                ckpt_path="./checkpoints/final_kto_checkpoint")