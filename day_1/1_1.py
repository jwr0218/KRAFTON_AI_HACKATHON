import torch
import torch.nn as nn
import torch.nn.functional as F

class MicroMultiplier(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(2, 1)
        self.emb.weight = nn.Parameter(torch.tensor([[0.0], [1.0]]))

        self.l1_attn_scale = nn.Parameter(torch.ones(6) * 10.0)
        
        # -----------------------------------------------------------
        # [핵심 버그 수정] NaN 방지를 위한 23번 Dummy Index 라우팅
        # -----------------------------------------------------------
        l1_mask = torch.zeros(6, 24, 24)
        l1_mask[:, :, 23] = 1.0  # 기본값: 모든 어텐션은 안전한 0.0 패딩 공간(23번)을 바라봄
        
        for c in range(6):
            for k in range(12):
                if 0 <= k - c <= 5:
                    l1_mask[c, 11+k, 23] = 0.0          # 유효한 쌍이 있으면 Dummy 해제
                    l1_mask[c, 11+k, c] = 1.0           # A_c
                    l1_mask[c, 11+k, 6+k-c] = 1.0       # B_{k-c}
        self.register_buffer('l1_mask', l1_mask)
        
        self.l1_w1 = nn.Parameter(torch.tensor([[2.0]]))
        self.l1_b1 = nn.Parameter(torch.tensor([-1.5]))
        self.l1_w2 = nn.Parameter(torch.tensor([[2.0]]))

        self.l2_attn_scale = nn.Parameter(torch.tensor([1.0]))
        
        l2_mask_sum = torch.zeros(24, 24)
        l2_mask_P = torch.zeros(24, 24)
        for k in range(12):
            for m in range(k):
                l2_mask_sum[11+k, 11+m] = 2.0 ** -(k - m)
                l2_mask_P[11+k, 12+m] = 2.0 ** -(k - m)
        self.register_buffer('l2_mask_sum', l2_mask_sum)
        self.register_buffer('l2_mask_P', l2_mask_P)
        
        self.l2_mlp_w1 = nn.Parameter(torch.ones(12, 1))
        self.l2_mlp_b1 = nn.Parameter(-torch.arange(12, dtype=torch.float32))
        w2_vals = [1.0] + [(-1)**k * 2.0 for k in range(1, 12)]
        self.l2_mlp_w2 = nn.Parameter(torch.tensor([w2_vals]))

        self.head_w = nn.Parameter(torch.tensor([[-2.0], [2.0]]))
        self.head_b = nn.Parameter(torch.tensor([1.0, -1.0]))

    def forward(self, x):
        batch_size, seq_len = x.size()
        v = self.emb(x)
        
        if seq_len < 24:
            pad = torch.zeros(batch_size, 24 - seq_len, 1, device=v.device)
            v_full = torch.cat([v, pad], dim=1)
        else:
            v_full = v

        scores = self.l1_mask * self.l1_attn_scale.view(6, 1, 1)
        scores = scores.masked_fill(self.l1_mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        
        head_outputs = torch.einsum('h i j, b j d -> b i h', attn, v_full)
        x_in = head_outputs.unsqueeze(-1)
        
        hidden = F.relu(F.linear(x_in, self.l1_w1, self.l1_b1))
        partial_sum = F.linear(hidden, self.l1_w2).sum(dim=2)

        carry_sum = torch.einsum('i j, b j d -> b i d', self.l2_mask_sum, partial_sum)
        carry_P = torch.einsum('i j, b j d -> b i d', self.l2_mask_P, v_full)
        carry = (carry_sum - carry_P) * self.l2_attn_scale
        
        total_sum = partial_sum + carry

        pulse_hidden = F.relu(F.linear(total_sum, self.l2_mlp_w1, self.l2_mlp_b1))
        out_bit = F.linear(pulse_hidden, self.l2_mlp_w2)

        logits = F.linear(out_bit, self.head_w, self.head_b)
        return logits[:, :seq_len, :]

if __name__ == '__main__':
    model = MicroMultiplier()
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Problem 1-1 파라미터 수 (P_1): {param_count} 개")
    
    import random
    print("\n--- 자기회귀 디코딩 테스트 시작 ---")
    
    for i in range(3):
        a, b = random.randint(0, 63), random.randint(0, 63)
        expected = a * b
        
        a_bin = [int(x) for x in format(a, '06b')[::-1]]
        b_bin = [int(x) for x in format(b, '06b')[::-1]]
        ctx = torch.tensor([a_bin + b_bin], dtype=torch.long)
        
        print(f"\nTest {i+1}: {a} * {b} = {expected}")
        
        generated = ctx
        with torch.no_grad():
            for step in range(12):
                logits = model(generated)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                
        pred_p = generated[:, 12:].squeeze().tolist()
        pred_value = sum(val * (2**idx) for idx, val in enumerate(pred_p))
        
        print(f"Generated Bits: {pred_p}")
        print(f"Decoded Value : {pred_value}  {'✅ SUCCESS' if pred_value == expected else '❌ FAIL'}")