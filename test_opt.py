import torch
import torch.nn as nn

p1 = torch.tensor([1.0, 2.0])
grad = torch.tensor([0.1, 0.2])
exp_avg = torch.tensor([0.01, 0.02])
exp_avg_sq = torch.tensor([0.001, 0.002])
step_t = torch.tensor(1.0)
lr_t = torch.tensor(0.1)
beta1_t = torch.tensor(0.9)
beta2_t = torch.tensor(0.999)
eps_t = torch.tensor(1e-8)
wd_t = torch.tensor(0.01)

bias1 = 1 - beta1_t ** step_t
bias2 = 1 - beta2_t ** step_t
denom1 = (exp_avg_sq / bias2).sqrt() + eps_t
step_size = lr_t / bias1

p2 = p1.clone()
denom2 = (exp_avg_sq / bias2).sqrt_().add_(eps_t)

print("denom1 == denom2:", torch.allclose(denom1, denom2))

p1_out = p1.clone()
p1_out.add_(exp_avg / denom1, alpha=-step_size)

p2_out = p2.clone()
p2_out.addcdiv_(exp_avg, denom2, value=-step_size)

print("p1_out == p2_out:", torch.allclose(p1_out, p2_out))
