import torch

def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
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

def get_data():
    p = torch.ones(10)
    grad = torch.ones(10) * 0.1
    exp_avg = torch.zeros(10)
    exp_avg_sq = torch.zeros(10)
    step_t = torch.tensor(1.0)
    lr_t = torch.tensor(0.01)
    beta1_t = torch.tensor(0.9)
    beta2_t = torch.tensor(0.999)
    eps_t = torch.tensor(1e-8)
    wd_t = torch.tensor(0.01)
    return p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t

p1, grad1, ea1, eas1, st1, lr1, b1_1, b2_1, eps1, wd1 = get_data()
adamw_step_fused(p1, grad1, ea1, eas1, st1, lr1, b1_1, b2_1, eps1, wd1)

p2, grad2, ea2, eas2, st2, lr2, b1_2, b2_2, eps2, wd2 = get_data()
adamw_step_fused_opt(p2, grad2, ea2, eas2, st2, lr2, b1_2, b2_2, eps2, wd2)

print("Match:", torch.allclose(p1, p2))
