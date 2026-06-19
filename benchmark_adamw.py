import torch
import time

def adamw_step_fused_orig(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

def adamw_step_fused_opt(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt_().add_(eps_t)
    step_size = lr_t / bias1
    p.addcdiv_(exp_avg, denom, value=-step_size)

N = 1000000
p = torch.randn(N, device='cpu')
grad = torch.randn(N, device='cpu')
exp_avg = torch.zeros(N, device='cpu')
exp_avg_sq = torch.zeros(N, device='cpu')

step_t = torch.tensor(1.0, device='cpu')
lr_t = torch.tensor(0.1, device='cpu')
beta1_t = torch.tensor(0.9, device='cpu')
beta2_t = torch.tensor(0.999, device='cpu')
eps_t = torch.tensor(1e-8, device='cpu')
wd_t = torch.tensor(0.01, device='cpu')

t0 = time.time()
for _ in range(100):
    adamw_step_fused_orig(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t)
t1 = time.time()
print(f"Orig: {t1-t0:.4f}s")

t0 = time.time()
for _ in range(100):
    adamw_step_fused_opt(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t)
t1 = time.time()
print(f"Opt: {t1-t0:.4f}s")
