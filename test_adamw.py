import ast
import torch

with open('train.py', 'r') as f:
    source = f.read()

parsed = ast.parse(source)

nodes_to_keep = []
for node in parsed.body:
    if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
        if 'prepare' not in [n.name for n in getattr(node, 'names', [])] and 'kernels' not in [n.name for n in getattr(node, 'names', [])]:
             nodes_to_keep.append(node)
    elif isinstance(node, ast.ClassDef) and node.name == 'MuonAdamW':
        nodes_to_keep.append(node)
    elif isinstance(node, ast.FunctionDef) and node.name in ('adamw_step_fused', 'muon_step_fused'):
        nodes_to_keep.append(node)
    elif isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None) == 'polar_express_coeffs':
        nodes_to_keep.append(node)

new_module_ast = ast.Module(body=nodes_to_keep, type_ignores=[])
code = compile(new_module_ast, filename="<ast>", mode="exec")

namespace = {}
exec(code, namespace)

MuonAdamW = namespace['MuonAdamW']

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.linear2 = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))

model = DummyModel()
param_groups = [
    {'kind': 'adamw', 'params': list(model.parameters()), 'lr': 0.1, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.01}
]
optimizer = MuonAdamW(param_groups)

loss = model(torch.randn(1, 10)).sum()
loss.backward()

optimizer.step()
print("Step successful!")
