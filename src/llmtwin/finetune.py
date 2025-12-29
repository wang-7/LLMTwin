import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset, concatenate_datasets
from typing import Any
from pydantic import BaseModel

def finetune(
        loadconfig: "LoadModelConfig",
        peftconfig: "PeftModelConfig",
        ):
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name='unsloth/Meta-Llama-3.1-8B',
        **loadconfig.model_dump(),
        )
    
    model = FastLanguageModel.get_peft_model(
        model=model,
        **peftconfig.model_dump(),
    )

    dataset1 = load_dataset("mlabonne/llmtwin", split="train")
    dataset2 = load_dataset("mlabonne/FineTome-Alpaca-100k", split="train[:10000]")
    dataset = concatenate_datasets([dataset1, dataset2])

    alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Response:
    {}"""
    EOS_TOKEN = tokenizer.eos_token
    def format_samples_sft(examples):
        text = []
        for instruction, output in zip(examples["instruction"], examples["output"], strict=False):
            message = alpaca_template.format(instruction, output) + EOS_TOKEN
            text.append(message)
        return {"text": text}

    dataset = dataset.map(format_samples_sft, batched=True, remove_columns=dataset.column_names)
    # dataset = dataset.map(format_samples_sft, batched=True,)
    dataset = dataset.train_test_split(test_size=0.05)
    tokenizer = get_chat_template(tokenizer, chat_template='chatml')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
            
    trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            dataset_text_field="text",
            max_seq_length=2048,
            packing=False,
            args=TrainingArguments(
                learning_rate=3e-4,
                average_tokens_across_devices=False,
                logging_steps=1,
                report_to='tensorboard',
                seed=0,
            )
        )


    trainer.train()

if __name__ == '__main__':

    class LoadModelConfig(BaseModel):
        max_seq_length: int = 2048
        dtype: Any | None = None # None for auto detection
        load_in_4bit: bool = False

    class PeftModelConfig(BaseModel):
        r: int = 16
        target_modules: Any = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        lora_alpha: int = 16
        lora_dropout: float = 0
        bias: str = "none"
    
    loadconfig = LoadModelConfig()
    peftconfig = PeftModelConfig()
    finetune(loadconfig, peftconfig)   