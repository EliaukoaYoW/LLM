from typing import Any, Dict, List
import torch
from loguru import logger

# 指令 + 回答 学 “如何按人类意图回答”
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

# Pretrain 数据集中的 Label 来自于自身 学“通用语言能力”
class PretrainCollator(object):
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = [x['input_ids'] for x in batch if x['input_ids'] is not None]
        # 找出 batch 中的最大长度
        lengths = [len(x) for x in batch]
        # 取出 batch 中的最大长度，如果超过 max_seq_length 则取 max_seq_length
        batch_max_len = min(max(lengths), self.max_seq_length)

        input_ids_batch, attention_mask_batch, labels_batch = [], [], []
        for x in batch:
            input_ids = x
            attention_mask = [1] * len(input_ids)
            padding_len = batch_max_len - len(input_ids)
            # 手动填充
            labels = input_ids + [-100] * padding_len
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            # 手动截断
            input_ids = input_ids[:self.max_seq_length]
            labels = labels[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]

            input_ids_batch.append(input_ids)
            labels_batch.append(labels)
            attention_mask_batch.append(attention_mask)

        # 将list转换为tensor，得到最终的的模型输入
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        labels_batch = torch.tensor(labels_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        inputs = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'labels': labels_batch
        }
        return inputs