import autograd.numpy as np
from autograd import grad

def function(x, y):
    return x**2 + y**2

# 计算梯度
grad_function = grad(function, argnum=(0, 1))

# 测试
x, y = 1.0, 2.0
df_dx, df_dy = grad_function(x, y)
print(f"Gradient at ({x}, {y}): ({df_dx}, {df_dy})")