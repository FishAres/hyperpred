# %%
from HyperNet import Hypernet, HyperFC
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
%load_ext autoreload
%autoreload 2


# %%
net = Hypernet(32, 64, 5)
# %%
x = torch.randn(32, 32, 10, 10)
# %%
net(x, x)
# %%


def f(x, g, r0, rmax=10):
    if x < 0:
        return r0 * np.math.tanh(g*x / r0)
    else:
        return (rmax - r0) * np.math.tanh(g*x / (rmax - r0))


# %%

x = np.arange(-5, 5, 0.1)
# %%

g = 1
r0 = 1
plt.plot(x, f(x, g, r0))
# %%
