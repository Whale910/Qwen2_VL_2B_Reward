import torch
import torch.nn as nn
import torch.nn.functional as F

class PairwiseRankingLoss(nn.Module):
    def __init__(self, beta=0.1, lb_weight=0.1, z_weight=1e-3, top_k=2):
        super().__init__()
        self.beta = beta
        self.lb_weight = lb_weight
        self.z_weight = z_weight
        self.top_k = top_k

    def forward(
            self,
            chosen_rewards,
            rejected_rewards,
            chosen_logits=None,
            ref_chosen_logits=None,
            attention_mask=None,
            router_logits=None  # [batch, num_experts]
    ):
        # Ranking Loss
        diff = chosen_rewards - rejected_rewards
        ranking_loss = -F.logsigmoid(diff).mean()

        # KL Loss
        kl_loss = torch.tensor(0.0, device=chosen_rewards.device)
        if chosen_logits is not None and ref_chosen_logits is not None:
            metrics_log_probs = F.log_softmax(chosen_logits, dim=-1)
            ref_probs = F.softmax(ref_chosen_logits, dim=-1)
            kl_div = F.kl_div(metrics_log_probs, ref_probs, reduction='none').sum(dim=-1)
            if attention_mask is not None:
                kl_loss = (kl_div * attention_mask).sum() / attention_mask.sum()
            else:
                kl_loss = kl_div.mean()

        # MoE 辅助 Loss
        lb_loss = torch.tensor(0.0, device=chosen_rewards.device)
        z_loss = torch.tensor(0.0, device=chosen_rewards.device)

        if router_logits is not None:
            # P_i: 平均路由概率
            flat_logits = router_logits.view(-1, router_logits.size(-1))
            probs = F.softmax(flat_logits, dim=-1)
            P_i = probs.mean(0)  # [num_experts]

            # f_i: 专家选择频率，每个专家在全量 token 中被选中的比例
            num_experts = probs.size(-1)
            _, top_k_indices = torch.topk(flat_logits, self.top_k, dim=-1)  # [tokens, top_k]

            # 使用 one-hot 标记选中的专家: [tokens, top_k, num_experts]
            expert_mask = F.one_hot(top_k_indices, num_experts).float()
            expert_mask_per_token = torch.max(expert_mask, dim=-2).values  # [tokens, num_experts]

            f_i = expert_mask_per_token.mean(0)  # [num_experts]

            # 负载均衡 Loss: N * Σ(f_i * P_i)
            lb_loss = num_experts * torch.sum(f_i * P_i)

            log_z = torch.logsumexp(flat_logits, dim=-1)
            z_loss = torch.mean(log_z ** 2)

        total_loss = (
                ranking_loss
                + self.beta * kl_loss
                + self.lb_weight * lb_loss
                + self.z_weight * z_loss
        )
        with torch.no_grad():
            accuracy = (chosen_rewards > rejected_rewards).float().mean()

        return total_loss, ranking_loss, kl_loss, lb_loss, z_loss, accuracy
