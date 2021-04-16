import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math


class LabelSmoothLoss(nn.Module): 
    def __init__(self, classes=5, smoothing=0.1, dim=-1): 
        super(LabelSmoothLoss, self).__init__() 
        self.confidence = 1.0 - smoothing 
        self.smoothing = smoothing 
        self.cls = classes 
        self.dim = dim 

    def forward(self, pred, target): 
        pred = pred.log_softmax(dim=self.dim) 
        with torch.no_grad():
            true_dist = torch.zeros_like(pred) 
            true_dist.fill_(self.smoothing / (self.cls - 1)) 
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) 

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
