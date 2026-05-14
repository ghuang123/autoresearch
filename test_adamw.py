import ast

def extract_code():
    with open('train.py', 'r') as f:
        tree = ast.parse(f.read())

    code = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ('adamw_step_fused', 'muon_step_fused'):
            code.append(ast.unparse(node))
        elif isinstance(node, ast.ClassDef) and node.name == 'MuonAdamW':
            code.append(ast.unparse(node))

    return '\n\n'.join(code)

code_str = extract_code()
code_str = code_str.replace("@torch.compile(dynamic=False, fullgraph=True)", "")

exec_env = {}
import torch
exec("import torch", exec_env)
exec(code_str, exec_env)

MuonAdamW = exec_env['MuonAdamW']

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(10, 10))

model = DummyModel()
loss = model.w.sum()
loss.backward()

optimizer = MuonAdamW([
    {'kind': 'adamw', 'params': [model.w], 'lr': 0.1, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.01}
])

for group in optimizer.param_groups:
    group['initial_lr'] = group['lr']

optimizer.step()

# Validate that parameters were actually updated and are not NaN
assert not torch.isnan(model.w).any()
assert not torch.allclose(model.w, torch.ones(10, 10))
print("Tests passed successfully.")
