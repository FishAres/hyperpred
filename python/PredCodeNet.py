# %%

import torch
import torch.nn as nn
import torch.nn.functional as F
from HyperNet import Hypernet, PredCodeNet

import numpy as np

# %%
%load_ext autoreload
%autoreload 2

# %%

net = PredCodeNet(36, 10, 32, 16, 0.1, 0.1)
# %%

x = torch.randn(10, 2, 1)

net(x)[0].shape
# %%

net(x.permute(0, 1, 2, 3))
# %%
x.permute(tuple(np.arange(-1, 3))).shape

# %%
tuple(np.arange(-1, 3))

# %%
