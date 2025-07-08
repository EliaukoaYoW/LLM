from loguru import logger
import os
from os.path import join
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from Compentent.collator import PretrainCollator, SFTDataCollator
from Compentent.model import Qwen3ForCausalLM
from Compentent.argument import CustomizedArguments
from Compentent.template import template_dict
from Compentent.dataset import UnifiedSFTDataset,ChatGLM2SFTDataset,ChatGLM3SFTDataset,UnifiedDPODataset
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    Trainer,
    AddedToken
)

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import Any, Dict, List
from loguru import logger

# 加载数据集
class MyDataset(Dataset):
    def __init__(self, file, tokenizer, max_seq_length, template):
        self.tokenizer = tokenizer

        self.template_name = template.template_name        # 模板名称
        self.system_format = template.system_format        # 系统提示 (Role)
        self.user_format = template.user_format            # 用户提示 (Question)
        self.assistant_format = template.assistant_format  # 助手回复 (Answer)
        self.system = template.system

        self.max_seq_length = max_seq_length
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logger.info(f'Use template "{self.template_name}" for training')
        logger.info("There are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # 每条数据拼接格式为: {system_format}{user_format}{assistant_format}{user_format}{assistant_format}...
        data = self.data_list[index]
        data = json.loads(data)
        input_ids, target_mask = [], []

        # 设置系统提示信息
        if self.system_format is not None:
            system = data['Instruction'].strip() if 'Instruction' in data.keys() else self.system
            # system 信息不为空
            if system is not None:
                system_text = self.system_format.format(content=system)
                input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
                target_mask = [0] * len(input_ids)


        human = data['Text'] + '\n' + data['Target']
        assistant = data['Stance'].strip()

        human = self.user_format.format(content=human, stop_token=self.tokenizer.eos_token)
        assistant = self.assistant_format.format(content=assistant, stop_token=self.tokenizer.eos_token)

        input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
        output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)

        input_ids += input_tokens + output_tokens
        target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)  # 问题 [Mask=0] 答案 [Mask=1]

        assert len(input_ids) == len(target_mask), "Input_ids_len != Target_mask_len "  # assert condition,message 不满足条件适发送消息
        # 对长度进行截断
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs

class SFTDataCollator(object):
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 找出 batch 中的最大长度
        lengths = [len(x['input_ids']) for x in batch if x['input_ids'] is not None]
        # 取出 batch中的最大长度 如果超过 max_seq_length 则取 max_seq_length
        batch_max_len = min(max(lengths), self.max_seq_length)
        # batch_max_len = self.max_seq_length

        input_ids_batch, attention_mask_batch, target_mask_batch = [], [], []
        # 截断或填充
        for x in batch:
            input_ids = x['input_ids']
            attention_mask = x['attention_mask']
            target_mask = x['target_mask']
            if input_ids is None:
                logger.info('some input_ids is None')
                continue
            padding_len = batch_max_len - len(input_ids)
            # 手动填充
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            target_mask = target_mask + [0] * padding_len
            # 手动截断
            input_ids = input_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
            target_mask = target_mask[:self.max_seq_length]

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            target_mask_batch.append(target_mask)

        # 将 List 转换为 Tensor 得到最终的的模型输入
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        target_mask_batch = torch.tensor(target_mask_batch, dtype=torch.long)

        labels = torch.where(target_mask_batch == 1, input_ids_batch, -100) # 当 target_mask == 1 时保留原位置上的inputs_ids 否则替换为-100
        inputs = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'labels': labels
        }
        return inputs


file = "Data/my_data.jsonl"
model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
print(model)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

template = template_dict['qwen']
logger.info('Loading data with MyDataset')
train_dataset = MyDataset(file, tokenizer=tokenizer,max_seq_length=1024,template=template)

logger.info('Loading data with SFTDataCollator')
data_collator = SFTDataCollator(tokenizer=tokenizer, max_seq_length=1024)
loader = DataLoader(train_dataset,batch_size=1,collate_fn=data_collator)


for index,data in enumerate(loader,start=0):
    if index==0:
        eos_token_id = tokenizer.eos_token_id  # 151645
        input_ids,attention_mask,labels = data['input_ids'],data['attention_mask'],data['labels']

        generated = input_ids
        print(tokenizer.decode(input_ids.tolist()[0],skip_special_tokens=True))

        output = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = output.logits  # shape: (batch_size, sequence, vocab_size)

        # 取概率最高的 token(greedy search)
        next_token = torch.argmax(logits[:,-1:], dim=-1, keepdim=True).squeeze(1)  # shape: (batch_size, 1)
        print(next_token.item())

        # 添加到已生成序列中
        generated = torch.cat([generated, next_token], dim=1)

        # 如果遇到 eos_token，提前结束
        if eos_token_id is not None and (next_token == eos_token_id).any():
            print("---------- End ----------")
            break

        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        print(texts)
    else:
        break
