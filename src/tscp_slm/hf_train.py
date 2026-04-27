from __future__ import annotations

import argparse

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from .slm_config import SLMTrainConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a TSCP SLM adapter on GPU.")
    parser.add_argument("--config", required=True, help="Path to training config JSON.")
    args = parser.parse_args()

    config = SLMTrainConfig.load(args.config)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype="auto",
        device_map="auto",
    )

    dataset = load_dataset(
        "json",
        data_files={
            "train": config.train_file,
            "validation": config.validation_file,
        },
    )

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        bf16=True,
        report_to=[],
        seed=config.seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="prompt",
        max_seq_length=config.max_seq_length,
        formatting_func=lambda example: [
            f"{prompt}\n{completion}"
            for prompt, completion in zip(example["prompt"], example["completion"])
        ],
        peft_config=peft_config,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


if __name__ == "__main__":
    main()
