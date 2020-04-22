from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


class LiftedStructureLoss(nn.Module):
    def __init__(self, alpha=40, beta=0, margin=0.4, epoch_num = 300, Dynamic_margin=None, sample_method=None, **kwargs):
        super(LiftedStructureLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.Dynamic_margin = Dynamic_margin
        self.sample_method = sample_method
        self.epoch_num = epoch_num

        self.pos_margin = 1
        self.neg_margin = 0
        

        if self.sample_method is not None:
            print("sample_method is on")
            self.neg_margin = 0.1
            self.pos_margin = 0.9
            print("pos_margin is:" + str(self.pos_margin))
            print("neg_margin is:" + str(self.neg_margin))

        if self.Dynamic_margin is not None:
            print("Dynamic_margin is on")

        print(self.margin)
        print(self.alpha)
        print(self.epoch_num)
    def forward(self, inputs, targets, nonorm):
        n = inputs.size(0)

        sim_mat = torch.matmul(inputs, inputs.t())

        targets = targets

        base = 0.5
        loss = list()
        c = 0

        for i in range(n):
            pos_pair_ = torch.masked_select(sim_mat[i], targets==targets[i])

            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < self.pos_margin)
            neg_pair_ = torch.masked_select(sim_mat[i], targets!=targets[i])

            if self.sample_method is not None:
                # pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < self.pos_margin)
                neg_pair_ = neg_pair_[neg_pair_ + 0.5 > min(pos_pair_)]
                neg_pair_ = torch.masked_select(neg_pair_, neg_pair_ > self.neg_margin)


            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_pair_)[0]


            if self.Dynamic_margin is not None:

                pos_pair = pos_pair_
                neg_pair = neg_pair_ 
            
                pos_loss = torch.log(torch.sum(torch.exp(1.0 - pos_pair + self.epoch_num / 300 * (pos_pair-self.pos_margin)**2)))
                neg_loss = torch.log(torch.sum(torch.exp(neg_pair + self.epoch_num / 300 * (self.neg_margin - neg_pair)**2)))



            else:  
                pos_pair = pos_pair_
                neg_pair = neg_pair_ 

                pos_loss = torch.log(torch.sum(torch.exp(1.0 - pos_pair)))
                neg_loss = torch.log(torch.sum(torch.exp(neg_pair)))

            if len(neg_pair) == 0:
                c += 1
                continue

            loss.append(pos_loss + neg_loss)
                    
        loss = sum(loss)/n
#        print(loss)
        prec = float(c)/n
        mean_neg_sim = torch.mean(neg_pair_).item()
        mean_pos_sim = torch.mean(pos_pair_).item()
        return loss, prec, mean_pos_sim, mean_neg_sim
        return loss, prec, mean_pos_sim, mean_neg_sim




