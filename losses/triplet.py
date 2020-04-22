from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


def similarity(inputs_):
    # Compute similarity mat of deep feature
    # n = inputs_.size(0)
    sim = torch.matmul(inputs_, inputs_.t())
    return sim


class TripletLoss(nn.Module):
    def __init__(self, beta=None, margin=0, epoch_num = 300, Dynamic_margin=True, **kwargs):
        super(TripletLoss, self).__init__()
        self.beta = beta
        self.margin = 0.5
        self.Dynamic_margin = Dynamic_margin
        self.epoch_num = epoch_num
        if self.Dynamic_margin is not None:
            print("Dynamic_margin is on")
        print(self.epoch_num)

    def forward(self, inputs, targets, nonorm):
        n = inputs.size(0)
        # Compute similarity matrix
        sim_mat = similarity(inputs)
        targets = targets.cuda()
        eyes_ = Variable(torch.eye(n, n)).cuda()
        pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        neg_mask = eyes_.eq(eyes_) - pos_mask
        pos_mask = pos_mask - eyes_.eq(1)

        pos_sim = torch.masked_select(sim_mat, pos_mask)
        neg_sim = torch.masked_select(sim_mat, neg_mask)
        num_instances = len(pos_sim)//n + 1
        num_neg_instances = n - num_instances   

        pos_sim = pos_sim.resize(len(pos_sim)//(num_instances-1), num_instances-1)

        neg_sim = neg_sim.resize(
            len(neg_sim) // num_neg_instances, num_neg_instances)
        #  clear way to compute the loss first
        loss = list()
        c = 0
        
        for i, pos_pair_ in enumerate(pos_sim):

            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_sim[i])[0]

            if self.Dynamic_margin is not None:

                for j, pos_pair_one in enumerate(pos_pair_):
                    if(pos_pair_one > 0.9):
                        continue
                    neg_pair_ = torch.masked_select(neg_pair_, neg_pair_ > 0.1)
                    neg_pair_ = torch.masked_select(neg_pair_, neg_pair_ > pos_pair_one - self.margin)
                    cha_pair = pos_pair_one - neg_pair_
                    if len(cha_pair) < 1:
                        c += 1
                        continue
                    cha_loss = self.margin - cha_pair + self.epoch_num / 300 * (0.9 - pos_pair_one)**2
                    cha_loss = torch.mean(cha_loss + self.epoch_num / 300 * (neg_pair_ - 0.1)**2)

                    loss.append(cha_loss)

            else:

                for j, pos_pair_one in enumerate(pos_pair_):
                    cha_pair = pos_pair_one - neg_pair_
                    cha_pair = torch.masked_select(cha_pair, cha_pair < self.margin)
                    if len(cha_pair) < 1:
                        c += 1
                        continue
                    cha_loss = torch.mean(self.margin - cha_pair)
                    loss.append(cha_loss)


        loss = sum(loss)/(n * (num_instances-1))
        prec = float(c)/(n * (num_instances-1))
        mean_neg_sim = torch.mean(neg_pair_).item()
        mean_pos_sim = torch.mean(neg_pair_).item()
        return loss, prec, mean_pos_sim, mean_neg_sim


def main():
    data_size = 32
    input_dim = 3
    output_dim = 2
    num_class = 4
    # margin = 0.5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    # print(x)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)
    y_ = 8*list(range(num_class))
    targets = Variable(torch.IntTensor(y_))

    print(HardMiningLoss()(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')


