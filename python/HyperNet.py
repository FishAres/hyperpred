import torch.nn as nn
import torch.nn.functional as F
import torch


class Hypernet(nn.Module):

    def __init__(self, K: int, H: int, Z: int):
        """Constructor

        Args:
            K (int): The number of neurons
            H (int): RNN hidden dimension
            Z (int): The number of mixture matrices
        """
        super(Hypernet, self).__init__()
        self.K = K
        self.H = H
        self.Z = Z
        self.rnn = nn.RNNCell(self.K, self.H)
        self.fc = nn.Sequential(
            nn.Linear(self.H, 500),
            nn.BatchNorm1d(500),
            nn.ELU(),
            nn.Linear(500, 250),
            nn.BatchNorm1d(250),
            nn.ELU(),
            nn.Linear(250, 250),
            nn.BatchNorm1d(250),
            nn.ELU(),
            nn.Linear(250, 100),
            nn.BatchNorm1d(100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ELU(),
            nn.Linear(100, self.Z),
        )

    def forward(self, r, h):
        h_t = self.rnn(r, h)
        return self.fc(h_t), h_t


class PredCodeNet(nn.Module):

    def __init__(self, K: int, N: int, Z: int, H: int, lr_r: float, lmda: float):
        """Constructor

        Args:
            K (int): The number of neurons
            N (int): The dimension of input
            Z (int): The number of mixture matrices
            H (int): RNN hidden dimension
            lr_r (float): Learning rate for ISTA
            lmda (float): Sparsity for ISTA
        """
        super(PredCodeNet, self).__init__()
        self.K = K
        self.N = N
        self.Z = Z
        self.H = H
        # image generative layer
        self.U = nn.Linear(self.K, self.N, bias=False)
        self.hypernet = Hypernet(self.K, self.H, self.Z)
        self.V = torch.randn(self.Z, self.K, self.K, requires_grad=True) * 0.01
        # learning parameters
        self.lr_r = lr_r
        self.lmda = lmda

    def forward(self, x):
        batchsize = x.size(0)
        T = x.size(1)
        # initialize r
        r = torch.zeros((batchsize, self.K),
                        device=self.U.weight.device, requires_grad=True)
        # initialize hypernet dynamic
        h = torch.zeros(batchsize, self.H, requires_grad=True,
                        device=self.U.weight.device)
        # go through time
        image_loss = 0
        temp_loss = 0
        for t in range(T):
            w, h = self.hypernet(r, h)
            V_w = torch.matmul(w, self.V.reshape(self.Z, -1)
                               ).reshape(batchsize, self.K, self.K)
            # prediction
            r_hat = F.relu(torch.bmm(V_w, r.unsqueeze(2))).squeeze()
            # fit to image
            r = self.ista(x[:, t, :], r_hat.clone().detach()).clone().detach()
            # losses
            image_loss += ((x[:, t, :] - self.U(r)) ** 2).sum()
            temp_loss += ((r - r_hat) ** 2).sum()
        return image_loss, temp_loss

    def ista(self, x: torch.tensor, r: torch.tensor):
        """ISTA steps for sparsification

        Args:
            x ([torch.tensor]): Input for reconstruction
            r ([torch.tensor]): Initialization of the code

        Returns:
            [torch.tensor]: the sparse code fitted to x
        """
        r.requires_grad_(True)
        converged = False
        # update R
        optim = torch.optim.SGD([{'params': r, "lr": self.lr_r}])
        # train
        while not converged:
            old_r = r.clone().detach()
            # prediction
            x_hat = self.U(r)
            # loss
            loss = ((x - x_hat) ** 2).sum()
            loss.backward()
            # update R in place
            optim.step()
            # print(r.grad)
            # zero grad
            optim.zero_grad()
            self.zero_grad()
            # prox
            r.data = self.soft_thresholding_(r, self.lmda)
            # convergence
            converged = torch.norm(r - old_r) / torch.norm(old_r) < 0.01
            #print(torch.norm(r - old_r) / torch.norm(old_r))
        return r

    def normalize(self):
        with torch.no_grad():
            self.U.weight.data = F.normalize(self.U.weight.data, dim=0)
            for z in range(self.Z):
                self.V[z].data = F.normalize(self.V[z].data, dim=1)

    @staticmethod
    def soft_thresholding_(r, lmda):
        with torch.no_grad():
            rtn = F.relu(F.relu(r - lmda) - F.relu(-r - lmda))
        return rtn.data

    def zero_grad(self):
        self.U.zero_grad()

# class HyperLayer(nn.Module):
#     def __init__(self, M, N):
#         super(HyperLayer, self).__init__()

#         self.linear = nn.Linear(M, N)
#         self.batchnorm = nn.BatchNorm1d(N)
#         self.elu = nn.ELU()

#     def forward(self, x):
#         x = self.linear(x)
#         x = self.batchnorm(x)
#         x = self.elu(x)

#         return x


# class HyperFC(nn.Sequential):
#     def __init__(self, N, H, Z, K=500, weight_scaling=0.75):
#         """
#         N (int): Number of RNN units
#         H (int): RNN hidden dimension
#         Z (int): Number of mixture matrices
#         K (int ; optional): first fully connected layer after RNN
#         weight_scaling (float ; optional): factor for progressively scaling layer size
#         """
#         super(HyperFC, self).__init__()

#         self.N = N
#         self.H = H
#         self.Z = Z

#         no_layers = 5  # placeholder
#         final_input_size = int((weight_scaling ** (no_layers)) * K)
#         curr_size = self.H
#         new_size = K

#         for i in range(no_layers):

#             new_size = int(weight_scaling ** (i) * K)

#             self.add_module("hyperlayer_{}".format(i),
#                             HyperLayer(curr_size, new_size))

#             curr_size = new_size

#         self.add_module("output", nn.Linear(final_input_size, self.Z))


# class Hypernet(nn.Module):

#     def __init__(self, N, H, Z, K=500, weight_scaling=0.5):
#         """
#         N (int): Number of RNN units
#         H (int): RNN hidden dimension
#         Z (int): Number of mixture matrices
#         K (int ; optional): first fully connected layer after RNN
#         weight_scaling (float ; optional): factor for progressively scaling layer size
#         """
#         super(Hypernet, self).__init__()

#         self.rnn = nn.RNNCell(N, H)
#         self.fc = HyperFC(N, H, Z, K=K, weight_scaling=weight_scaling)

#     def forward(self, r, h):
#         h_t = self.rnn(r, h)
#         out = self.fc(h_t)
#         return out, h_t
