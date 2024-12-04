# train_online_dpo.py
from datasets import load_dataset
from trl import OnlineDPOConfig, OnlineDPOTrainer, PairRMJudge, AllTrueJudge
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.judges import BinaryGTJudge, BinaryDiffJudge, BinaryCorrectionJudge, BinaryDifficultyJudge
from src.online_cdpo_trainer import OnlineCDPOTrainer

import torch

rootpath = "/apdcephfs_qy3/share_301812049/"
train_dataset_file = "/apdcephfs_qy3/share_301812049/jianhuipang/project_o1/onlinerl/datasets/math500.train.trl.json"
model_path = "/apdcephfs_qy3/share_301812049/shared/model/Qwen/Qwen2.5-0.5B-Instruct"
# model_path = rootpath + "shared/model/Qwen/Qwen2.5-Math-7B"

train_dataset = load_dataset("json", data_files = train_dataset_file)["train"]

# print(train_dataset)
# print(train_dataset["train"][0])

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

judge = AllTrueJudge([BinaryGTJudge()])

training_args = OnlineDPOConfig(output_dir="Qwen2.5-Math-7B-OnlineDPO", logging_steps=1)
training_args.max_new_tokens = 256
training_args.per_device_train_batch_size = 1
training_args.deepspeed="/apdcephfs_qy3/share_301812049/jianhuipang/LLMs4MT/train/deepspeed_config_bf16.json"
training_args.bf16=True
training_args.gradient_checkpointing=True
training_args.save_strategy="steps"
training_args.save_steps=50

# model.gradient_checkpointing_enable()
# if hasattr(model, "enable_input_require_grads"):
#     model.enable_input_require_grads()
# else:
#     def make_inputs_require_grad(module, input, output):
#          output.requires_grad_(True)

#     model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

trainer = OnlineCDPOTrainer(
    model=model, judge=judge, args=training_args, processing_class=tokenizer, train_dataset=train_dataset, sample_k=2
)
trainer.train()
