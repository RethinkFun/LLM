from accelerate import PartialState
from datasets import load_dataset
from peft import TaskType, LoraConfig, get_peft_model
from transformers import Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dataclasses import dataclass, field
import transformers
from itertools import chain
import torch
import warnings

warnings.filterwarnings("ignore")


@dataclass
class CustomArguments(transformers.TrainingArguments):
    # LoRA_r
    lora_r: int = field(default=8)
    # 数据处理时的并行进程数
    num_proc: int = field(default=1)
    # 最大序列长度
    max_seq_length: int = field(default=32)
    # 验证策略，如不想进行验证，可以设置为 ‘no’
    eval_strategy: str = field(default="steps")
    # 每多少步进行一次验证
    eval_steps: int = field(default=100)
    # 随机种子
    seed: int = field(default=0)
    # 优化器
    optim: str = field(default="adamw_torch")
    # 训练epoch数
    num_train_epochs: int = field(default=2)
    # 每个设备上的批量大小
    per_device_train_batch_size: int = field(default=1)

    # 学习率
    learning_rate: float = field(default=5e-5)
    # 权重衰减
    weight_decay: float = field(default=0)
    # 预热步数
    warmup_steps: int = field(default=10)
    # 学习率规划期类型
    lr_scheduler_type: str = field(default="linear")
    # 是否使用梯度检查点
    gradient_checkpointing: bool = field(default=False)
    # 是否使用bf16作为混合精度训练类型
    bf16: bool = field(default=True)
    # 梯度累加步数
    gradient_accumulation_steps: int = field(default=1)

    # 日志记录的步长频率
    logging_steps: int = field(default=3)
    # checkpoint保存策略
    save_strategy: str = field(default="steps")
    # checkpoint保存的步长频率
    save_steps: int = field(default=3)
    # 总的保存checkpoint的数量
    save_total_limit: int = field(default=2)


parser = transformers.HfArgumentParser(CustomArguments)
training_args, = parser.parse_args_into_dataclasses()

model_path = '/data04/llama3/Meta-Llama-3.1-8B-Instruct'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    low_cpu_mem_usage=True,
    quantization_config=bnb_config,
    device_map={"": PartialState().process_index}
)
peft_config = LoraConfig(
    r=training_args.lora_r,
    target_modules=["q_proj",
                    "v_proj",
                    "k_proj",
                    "o_proj",
                    "gate_proj",
                    "down_proj",
                    "up_proj"
                    ],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=16,
    lora_dropout=0.05
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

train_dataset = load_dataset("text", data_dir="/home/xuepeng/pretrain_test/train_data", split="train")
eval_dataset = load_dataset("text", data_dir="/home/xuepeng/pretrain_test/eval_data", split="train")


def tokenization(example):
    return tokenizer(example["text"])


with training_args.main_process_first(desc="dataset map tokenization"):
    train_dataset = train_dataset.map(tokenization, remove_columns=["text"], num_proc=training_args.num_proc)
    eval_dataset = eval_dataset.map(tokenization, remove_columns=["text"], num_proc=training_args.num_proc)


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // training_args.max_seq_length) * training_args.max_seq_length
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + training_args.max_seq_length] for i in range(0, total_length, training_args.max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


with training_args.main_process_first(desc="dataset map tokenization"):
    train_dataset = train_dataset.map(group_texts, num_proc=training_args.num_proc, batched=True)
    eval_dataset = eval_dataset.map(group_texts, num_proc=training_args.num_proc, batched=True)

if __name__ == '__main__':
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    trainer.train()
    trainer.save_model("/data04/xuepeng/test_train")
