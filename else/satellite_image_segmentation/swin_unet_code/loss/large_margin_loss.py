import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

'''
proposed in the BMVC2019 paper: [Large Margin in Softmax Cross-Entropy Loss
link to paper](https://staff.aist.go.jp/takumi.kobayashi/publication/2019/BMVC2019.pdf)
'''

##
# version 1: use torch.autograd
class LargeMarginSoftmaxV1(nn.Module):

    def __init__(self, lam=0.3, reduction='mean', ignore_index=255, weight=None):
        super(LargeMarginSoftmaxV1, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lam = lam
        self.ce_crit = nn.CrossEntropyLoss(
                reduction='none', ignore_index=ignore_index, weight=weight)


    def forward(self, logits, label):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LargeMarginSoftmaxV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        logits = logits.float()
        logits.retain_grad()
        logits.register_hook(lambda grad: grad)
        with torch.no_grad():
            num_classes = logits.size(1)
            coeff = 1. / (num_classes - 1.)
            lb = label.clone().detach()
            mask = label == self.ignore_index
            lb[mask] = 0
            idx = torch.zeros_like(logits).scatter_(1, lb.unsqueeze(1), 1.)

        lgts = logits - idx * 1.e6
        q = lgts.softmax(dim=1)
        q = q * (1. - idx)

        log_q = lgts.log_softmax(dim=1)
        log_q = log_q * (1. - idx)
        mg_loss = ((q - coeff) * log_q) * (self.lam / 2)
        mg_loss = mg_loss * (1. - idx)
        mg_loss = mg_loss.sum(dim=1)

        ce_loss = self.ce_crit(logits, label)
        loss = ce_loss + mg_loss
        loss = loss[mask == 0]

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss


##
# version 2: user derived grad computation
class LargeMarginSoftmaxV2(nn.Module):

    def __init__(self, lam=0.3, reduction='mean', ignore_index=255):
        super(LargeMarginSoftmaxV2, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lam = lam

    def forward(self, logits, labels):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LargeMarginSoftmaxV2()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        logits = logits.float()
        mask = labels == self.ignore_index
        lb = labels.clone().detach()
        lb[mask] = 0
        loss = LargeMarginSoftmaxFuncV2.apply(logits, lb, self.lam)
        loss = loss[mask == 0]
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss



class LargeMarginSoftmaxFuncV2(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, labels, lam=0.3):
        num_classes = logits.size(1)
        coeff = 1. / (num_classes - 1.)
        idx = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1.)

        lgts = logits.clone()
        lgts[idx.bool()] = -1.e6
        q = lgts.softmax(dim=1)
        log_q = lgts.log_softmax(dim=1)
        losses = q.sub_(coeff).mul_(log_q).mul_(lam / 2.)
        losses[idx.bool()] = 0

        losses = losses.sum(dim=1).add_(F.cross_entropy(logits, labels, reduction='none'))

        ctx.variables = logits, labels, idx, coeff, lam
        return losses

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        '''
        compute gradient
        '''
        logits, labels, idx, coeff, lam = ctx.variables
        num_classes = logits.size(1)

        p = logits.softmax(dim=1)
        lgts = logits.clone()
        lgts[idx.bool()] = -1.e6
        q = lgts.softmax(dim=1)
        qx = q * lgts
        qx[idx.bool()] = 0

        grad = qx + q - q * qx.sum(dim=1).unsqueeze(1) - coeff
        grad = grad * lam / 2.
        grad[idx.bool()] = -1
        grad = grad + p

        grad.mul_(grad_output.unsqueeze(1))

        return grad, None, None



