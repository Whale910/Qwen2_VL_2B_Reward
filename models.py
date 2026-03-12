import torch
import torch.nn as nn
from transformers import Qwen2VLModel, Qwen2VLPreTrainedModel, Qwen2VLForConditionalGeneration
import torch.nn.functional as F

class MoEScoreHead(nn.Module):
    def __init__(self, hidden_size, num_experts=4, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.GELU(),
                nn.Linear(hidden_size // 4, 1, bias=False)
            ) for _ in range(num_experts)
        ])

        self.gate = nn.Linear(hidden_size, num_experts)
        self.noise_linear = nn.Linear(hidden_size, num_experts)

    def forward(self, x):
        # x: [batch_size, hidden_size]

        logits = self.gate(x)

        if self.training:
            noise = torch.randn_like(logits) * F.softplus(self.noise_linear(x))
            logits = logits + noise

        top_k_logits, top_k_indices = torch.topk(logits, self.k, dim=-1)

        weights = F.softmax(top_k_logits, dim=-1)

        batch_size = x.shape[0]
        final_scores = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)

        for i in range(self.k):
            exp_idx = top_k_indices[:, i]
            w = weights[:, i].unsqueeze(1)

            # 找出本轮 top-i 选中的所有样本，按专家分组运行
            for e_idx in range(self.num_experts):
                mask = (exp_idx == e_idx)
                if mask.any():
                    # 一次性运行该专家的所有相关样本
                    expert_outputs = self.experts[e_idx](x[mask])
                    final_scores[mask] += w[mask] * expert_outputs

        return final_scores, logits


class Qwen2VLRewardModel(nn.Module):
    def __init__(self, model_path, **kwargs):
        super().__init__()

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            **kwargs
        )

        self.config = self.model.config
        self.hidden_size = self.config.hidden_size

        # MLP架构
        # self.score_head = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.hidden_size),
        #     nn.GELU(),
        #     nn.Linear(self.hidden_size, 1)
        # )
        # for m in self.score_head.modules():
        #     if isinstance(m, nn.Linear):
        #         # 使用 Xavier 正态分布初始化权重，保持方差一致
        #         nn.init.xavier_normal_(m.weight)
        #         if m.bias is not None:
        #             # 偏置初始化为 0，防止初始分数值过大
        #             nn.init.zeros_(m.bias)

        self.score_head = MoEScoreHead(self.model.config.hidden_size, num_experts=4, k=2)

        for name, module in self.score_head.named_modules():
            if isinstance(module, nn.Linear):
                if 'gate' in name:
                    nn.init.normal_(module.weight, std=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

                elif 'noise_linear' in name:
                    nn.init.zeros_(module.weight)
                    nn.init.constant_(module.bias, -5.0)

                else:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        if hasattr(self.model, "device"):
            self.score_head.to(self.model.device)
            self.score_head.to(self.model.dtype)

        # 确保score_head的参数可训练
        for param in self.score_head.parameters():
            param.requires_grad = True

    def forward(
            self,
            input_ids,
            attention_mask=None,
            pixel_values=None,
            image_grid_thw=None,
            **kwargs
    ):
        """
        前向传播：接收图像和文本，输出得分
        """
        outputs = self.model(
            input_ids=input_ids,  # embedding 索引
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = outputs.logits

        # 取最后一层的隐藏状态: [batch_size, sequence_length, hidden_size]
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        else:
            # 这里的 [-1] 取出的是最后一层 Transformer 层输出的张量
            hidden_states = outputs.hidden_states[-1]

        # 在处理一个 batch 时句子长短不一，会有 padding，需要定位每个样本最后一个有效 Token (EOS token)
        if attention_mask is not None:
            # 找到 attention_mask 中最后一个 1 的位置
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = input_ids.shape[0]
            # 提取每个序列最后一个有效 token 的特征: [batch_size, hidden_size]
            last_token_hidden_states = hidden_states[
                torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        else:
            # 如果没有 padding (batch_size=1)，直接取最后一个 token
            last_token_hidden_states = hidden_states[:, -1, :]

        # scores = self.score_head(last_token_hidden_states) # MLP用
        scores, gate_logits = self.score_head(last_token_hidden_states)

        # MLP输出
        # return {
        #     "scores": scores,
        #     "logits": logits
        # }
        return {
            "scores": scores,
            "logits": logits,
            "router_logits": gate_logits
        }