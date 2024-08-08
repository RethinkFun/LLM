import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from peft import TaskType, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LlamaConfig, LlamaForCausalLM,LlamaModel
import torch

model_path = '/data04/llama3/Meta-Llama-3.1-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_path)

# # 1.从零开始训练大模型
# config = LlamaConfig() # 创建一个默认的Llama config
# config.num_hidden_layers = 12 # 配置网络结构
# config.hidden_size = 1024
# config.intermediate_size = 4096
# config.num_key_value_heads = 8
# # 用配置文件初始化一个大模型
# model = LlamaForCausalLM(config)

# 2.加载一个预训练的大模型

# # 4bit load
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    low_cpu_mem_usage=True,
    quantization_config=bnb_config
)

# 构造Lora模型
peft_config = LoraConfig(
        r=8,
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

model.to("cuda")
optimizer = torch.optim.AdamW(model.parameters())

text = "今天天气不错。"
input = tokenizer(text, return_tensors="pt")
input = {k: v.to("cuda") for k, v in input.items()}

#设置labels和inputs一致
input["labels"] = input["input_ids"].clone()

output = model(**input)

#获取模型的loss
loss = output.loss
loss.backward()
optimizer.step()
optimizer.zero_grad()

#保存模型
model.save_pretrained("output_dir")