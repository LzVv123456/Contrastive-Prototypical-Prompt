import numpy as np
import time
import random
import torch
from torch.utils.data import Dataset
from torch.distributions import normal


class ProtoDataset(Dataset):
    def __init__(self, args, prototypes, prototypes_var, classes):
        self.args = args
        self.prototypes = prototypes
        self.prototypes_var = prototypes_var
        self.classes = classes
        assert len(self.prototypes) == len(self.prototypes_var) == len(classes)

        if self.args.use_mc_proto:
            assert type(self.prototypes) == list
            assert type(self.prototypes_var) == list
            index_mapping = []
            for idx in range(len(classes)):
                cur_protos = self.prototypes[idx].squeeze()
                cur_mapping = torch.full((len(cur_protos), 1), idx)
                index_mapping.append(cur_mapping)
            self.prototypes = torch.cat(self.prototypes, dim=0).cuda()
            self.prototypes_var = torch.cat(self.prototypes_var, dim=0).cuda()
            self.index_mapping = torch.cat(index_mapping, dim=0).squeeze().cuda()
        else:
            assert type(self.prototypes) == list
            assert type(self.prototypes_var) == list
            self.prototypes = torch.stack(self.prototypes, dim=0).cuda()
            self.prototypes_var = torch.stack(self.prototypes_var, dim=0).cuda()

        self.scale = torch.sqrt(self.prototypes_var)  # (proto_num, embeding_dim)
        assert len(self.scale) == len(self.prototypes)


    def __len__(self):
        return self.prototypes.size(0)


    def __getitem__(self, idx):
        if self.args.proto_trans:
            proto_aug = self.proto_transform(idx)
        else:
            proto_aug = self.prototypes[idx]
        if self.args.use_mc_proto:
            label = self.classes[self.index_mapping[idx]]
        else:
            label = self.classes[idx]
        return proto_aug, label
        

    def proto_transform(self, idx):  
        proto = self.prototypes[idx]
        gaussian_noise = torch.normal(torch.zeros(len(proto)), 1).cuda() * self.scale[idx]
        proto_aug = proto + gaussian_noise
        return proto_aug

  


    