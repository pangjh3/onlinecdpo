# train_online_dpo.py
from datasets import load_dataset
from trl import OnlineDPOConfig, OnlineDPOTrainer, PairRMJudge
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.import_utils import is_llm_blender_available
from accelerate import Accelerator

if is_llm_blender_available():
    import llm_blender

model_path = "/apdcephfs_qy3/share_301812049/shared/model/Qwen/Qwen2.5-0.5B-Instruct"
data_path = "/apdcephfs_qy3/share_301812049/jianhuipang/project_o1/onlinerl/datasets/ultrafeedback-prompt"

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


class mypj(PairRMJudge):
    def __init__(self):
        if not is_llm_blender_available():
            raise ValueError("llm-blender is not installed. Please install it with `pip install llm-blender`.")
        self.blender = llm_blender.Blender()
        self.blender.loadranker("/apdcephfs_qy3/share_301812049/jianhuipang/project_o1/onlinerl/model/PairRM", device=Accelerator().device)

judge = mypj()
train_dataset = load_dataset(data_path, split="train")

training_args = OnlineDPOConfig(output_dir="Qwen2-0.5B-OnlineDPO", logging_steps=10)
trainer = OnlineDPOTrainer(
    model=model, judge=judge, args=training_args, processing_class=tokenizer, train_dataset=train_dataset
)
trainer.train()
