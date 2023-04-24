import os
import tqdm

os.environ['HF_DATASETS_CACHE'] = "cache"
os.environ['HUGGINGFACE_HUB_CACHE '] = "cache"
os.environ['TRANSFORMERS_CACHE'] = "cache"
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

from torch import nn
from transformers import TrainingArguments, Trainer, logging
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import bitsandbytes as bnb
from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader

from Train.data import Zhihu


tokenizer = AutoTokenizer.from_pretrained("/home/nlper_data/kuangzh/models/YeungNLP/firefly-1b4")
model = AutoModelForCausalLM.from_pretrained("/home/nlper_data/kuangzh/models/YeungNLP/firefly-1b4").to("cuda")

logging.set_verbosity_error()


def preprocess(row):
    input_texts = [t1 + t2 for t1, t2 in zip(row["title"], row['response'])]
    return tokenizer(input_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")


dataset = Zhihu.build_dataset()
dataset = dataset.map(preprocess, batched=True, batch_size=32, num_proc=8)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])


def compute_loss(inputs, outputs, return_outputs=False):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs['logits'][:, :-1].transpose(-1, -2), inputs['input_ids'][:, 1:])
    return (loss, outputs) if return_outputs else loss


default_args = {
    "output_dir": "tmp",
    "evaluation_strategy": "steps",
    "num_train_epochs": 1,
    "log_level": "error",
    "report_to": "none",
    "learning_rate": 1e-6,
    "gradient_accumulation_steps": 16,
    "per_device_train_batch_size": 8,
    "gradient_checkpointing": True
}


training_args = TrainingArguments(**default_args)
dataloader = DataLoader(dataset, batch_size=training_args.per_device_train_batch_size)
model.gradient_checkpointing_enable() if training_args.gradient_checkpointing else None

accelerator = Accelerator(mixed_precision='fp16')
adam_bnb_optim = bnb.optim.Adam8bit(
    model.parameters(),
    betas=(training_args.adam_beta1, training_args.adam_beta2),
    eps=training_args.adam_epsilon,
    lr=training_args.learning_rate,
)
model, optimizer, dataloader = accelerator.prepare(model, adam_bnb_optim, dataloader)


model.train()
p_bar = tqdm.tqdm(dataloader)
for step, batch in enumerate(p_bar, start=1):
    outputs = model(**batch).loss
    loss = compute_loss(batch, outputs)
    loss = loss / training_args.gradient_accumulation_steps
    accelerator.backward(loss)
    p_bar.set_postfix(loss=loss.item())
    if step % training_args.gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
