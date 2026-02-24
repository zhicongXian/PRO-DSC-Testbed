import torch
import torch.nn as nn

class TotalCodingRate(nn.Module):
    """ from https://github.com/ryanchankh/mcr2
    """
    def __init__(self, eps=0.01):
        super(TotalCodingRate, self).__init__()
        self.eps = eps
        
    def compute_discrimn_loss(self, W):
        """Discriminative Loss."""
        p, m = W.shape  #[d, B]
        I = torch.eye(p,device=W.device)
        scalar = p / (m * self.eps)
        sign, logdet =  torch.linalg.slogdet(I + scalar * W.matmul(W.T)) #torch.logdet(I + scalar * W.matmul(W.T))
        if sign <= 0:
            print("Matrix not positive definite!")
        return logdet / 2.
    
    def forward(self,X):
        return - self.compute_discrimn_loss(X.T)