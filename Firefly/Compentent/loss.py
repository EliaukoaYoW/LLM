import torch
import torch.nn as nn


class Loss(object):
    """
    所有 Loss 的类父类
    """
    def __call__(self, model, inputs, training_args, return_outputs=False):
        """
        todo label smoothing
        用于计算loss。
        看源码发现，return_outputs=True为train时调用，return_outputs=False为eval和predict调用
        :param model: 模型
        :param inputs: 模型输入，dict
        :param training_args: 训练配置参数
        :param return_outputs:是否返回模型的输出
        :return:
        """
        raise NotImplemented


class TargetLMLoss(Loss):

    def __init__(self, ignore_index):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def __call__(self, model, inputs, training_args, return_outputs=False):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        target_mask = inputs['target_mask']

        # 模型前馈预测
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]

        # 将 labels 中不属于 target 的部分 设为 ignore_index 只计算 target部分的 loss
        labels = torch.where(target_mask == 1, input_ids, self.ignore_index) # input_ids if target_mask==1 else -100
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return (loss, outputs) if return_outputs else loss