import torch
from torch import nn
from torch.nn import functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, args, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.args = args
        self.temperature = args.temp
        self.contrast_mode = contrast_mode
        self.base_temperature = args.temp
        self.eps = 1e-5

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
 
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() 

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # compute log_prob
        if self.args.supcon_wo_uniformity:
            n_mask = torch.ones_like(mask) - mask  

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        if self.args.supcon_wo_uniformity:
            exp_logits = torch.exp(logits) * n_mask
        else:
            exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + self.eps)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + self.eps)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        assert not torch.isnan(loss), print(loss)

        return loss


class ProtoConLoss(nn.Module):
    def __init__(self, args, prototypes, classes_up2now, proto_dataloader):
        super(ProtoConLoss, self).__init__()
        self.args = args
        self.prototypes = prototypes
        self.classes_up2now = classes_up2now
        self.proto_dataloader = proto_dataloader
        self.eps = 1e-5

        # normalize prototypes
        if type(self.prototypes) == list:
            self.prototypes = [F.normalize(proto, dim=-1, p=2) for proto in self.prototypes]
        else:
            self.prototypes = F.normalize(self.prototypes, dim=-1, p=2)

    def forward(self, features, labels):
        """
        features: (bs, dim)
        """
        assert len(features.size()) == 2
        labels = labels.unsqueeze(1)
        sample_num = len(features)

        # add pos proto
        if self.args.add_pos_proto:
            unique_labels = torch.unique(labels)
            unique_labels = unique_labels[unique_labels>=0]
            pos_proto, pos_label = [], []
            for label in unique_labels:
                if len(self.prototypes[label].size()) == 1:
                    proto = self.prototypes[label].unsqueeze(0)
                    pos_proto.append(proto)
                    tem_label = torch.full((len(proto), 1), label)
                else:
                    pos_proto.append(self.prototypes[label])
                    tem_label = torch.full((len(self.prototypes[label]), 1), label)
                pos_label.append(tem_label)
            pos_proto = torch.cat(pos_proto, dim=0)
            pos_label = torch.cat(pos_label, dim=0).cuda().long()
            features = torch.cat([features, pos_proto], dim=0)
            labels = torch.cat([labels, pos_label], dim=0)

        # add neg proto
        if self.args.add_neg_proto and self.classes_up2now is not None:
            for proto_samples, proto_labels in self.proto_dataloader:
                n_protos = F.normalize(proto_samples, dim=-1, p=2).cuda()
                n_proto_labels = proto_labels.cuda()
                nan_mask = torch.sum(torch.isnan(n_protos), axis=1)
                n_protos = n_protos[nan_mask==0, :]
                n_proto_labels = n_proto_labels[nan_mask==0]
                break

            features = torch.cat([features, n_protos], dim=0)
            labels = torch.cat([labels, n_proto_labels.unsqueeze(1)], dim=0)

        # compute logits
        dot_prodcut = torch.matmul(features, features.T)
        temped_contrast_num = torch.div(dot_prodcut, self.args.temp)
        temped_contrast_den = torch.div(dot_prodcut, self.args.neg_temp)
        # for numerical stability
        logits_max_num, _ = torch.max(temped_contrast_num, dim=1, keepdim=True)
        logits_num = temped_contrast_num - logits_max_num.detach()
        logits_max_den, _ = torch.max(temped_contrast_den, dim=1, keepdim=True)
        logits_den = temped_contrast_den - logits_max_den.detach()
        # generate mask
        mask = torch.eq(labels, labels.T).float().cuda()
        # logits_mask
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(len(labels)).view(-1, 1).cuda(),
            0
        )
        p_mask = mask * logits_mask
        # get negative mask
        if self.args.add_uniformity:
            n_mask = logits_mask
        else:
            n_mask = torch.ones_like(mask) - mask

        exp_logits_den = torch.exp(logits_den) * n_mask
        log_prob = logits_num - torch.log(exp_logits_den.sum(1, keepdim=True) + self.eps)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (p_mask * log_prob).sum(1) / (p_mask.sum(1) + self.eps)
        # loss
        loss = - mean_log_prob_pos.mean()
        return loss