from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, num_classes=5):
        super(SupConLoss, self).__init__()
        self.T = temperature
        self.num_classes = num_classes

    def forward(self, q, labels):
        device = (torch.device('cuda')
                  if q.is_cuda
                  else torch.device('cpu'))
        batch_size = q.shape[0]
        targets = labels.contiguous().view(-1, 1)
        # batch_cls_count = torch.eye(self.num_classes)[targets].sum(dim=0).squeeze()

        mask = torch.eq(targets, targets.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # class-complement
        logits = q.mm(q.T)
        logits = torch.div(logits, self.T)

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # class-averaging
        exp_logits = torch.exp(logits) * logits_mask
        # per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(
        #     batch_size, batch_size) - mask
        # exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)
        exp_logits_sum = exp_logits.sum(dim=1, keepdim=True)

        log_prob = logits - torch.log(exp_logits_sum + 1e-12)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.mean()
        return loss


class SupConPLoss(nn.Module):
    def __init__(self, temperature=0.07, num_classes=5):
        super(SupConPLoss, self).__init__()
        self.T = temperature
        self.num_classes = num_classes

    def forward(self, q, labels, prototypes):
        device = (torch.device('cuda')
                  if q.is_cuda
                  else torch.device('cpu'))
        batch_size = q.shape[0]
        targets = labels.contiguous().view(-1, 1)
        targets_prototypes = torch.arange(self.num_classes, device=device).view(-1, 1)
        targets = torch.cat([targets, targets_prototypes], dim=0)
        batch_cls_count = torch.eye(self.num_classes)[targets].sum(dim=0).squeeze()

        mask = torch.eq(targets[:batch_size], targets.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # class-complement
        q = torch.cat([q, prototypes], dim=0)
        logits = q[:batch_size].mm(q.T)
        logits = torch.div(logits, self.T)

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # class-averaging
        exp_logits = torch.exp(logits) * logits_mask
        per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(
            batch_size, batch_size + self.num_classes) - mask
        exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)
        # exp_logits_sum = exp_logits.sum(dim=1, keepdim=True)

        # log_prob = logits - torch.log(exp_logits_sum + 1e-12)
        log_prob = logits - torch.log(exp_logits_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.mean()
        return loss


class SupConPDALoss(nn.Module):  # data augmentation
    def __init__(self, temperature=0.07, num_classes=5):
        super(SupConPDALoss, self).__init__()
        self.T = temperature
        self.num_classes = num_classes

    def forward(self, q, labels, prototypes):
        device = (torch.device('cuda')
                  if q.is_cuda
                  else torch.device('cpu'))
        batch_size = labels.shape[0]
        targets = labels.contiguous().view(-1, 1)
        targets_prototypes = torch.arange(self.num_classes, device=device).view(-1, 1)
        targets = torch.cat([targets.repeat(2, 1), targets_prototypes], dim=0)
        batch_cls_count = torch.eye(self.num_classes)[targets].sum(dim=0).squeeze()

        mask = torch.eq(targets[:2 * batch_size], targets.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(2 * batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # class-complement
        q = torch.cat([q, prototypes], dim=0)
        logits = q[:2 * batch_size].mm(q.T)
        logits = torch.div(logits, self.T)

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # class-averaging
        exp_logits = torch.exp(logits) * logits_mask
        per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(
            2 * batch_size, 2 * batch_size + self.num_classes) - mask
        exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)
        # exp_logits_sum = exp_logits.sum(dim=1, keepdim=True)

        # log_prob = logits - torch.log(exp_logits_sum + 1e-12)
        log_prob = logits - torch.log(exp_logits_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.view(2, batch_size).mean()
        return loss


class TargetedSupConPDALoss(nn.Module):  # data augmentation
    def __init__(self, temperature=0.07, num_classes=5):
        super(TargetedSupConPDALoss, self).__init__()
        self.T = temperature
        self.num_classes = num_classes

    def forward(self, q, labels, prototypes, prototypes_labels):
        device = (torch.device('cuda')
                  if q.is_cuda
                  else torch.device('cpu'))
        batch_size = labels.shape[0]
        targets = labels.contiguous().view(-1, 1)
        # targets_prototypes = torch.arange(self.num_classes, device=device).view(-1, 1)
        targets = torch.cat([targets.repeat(2, 1), prototypes_labels], dim=0)
        batch_cls_count = torch.eye(self.num_classes)[targets].sum(dim=0).squeeze()

        mask = torch.eq(targets[:2 * batch_size], targets.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(2 * batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # class-complement
        q = torch.cat([q, prototypes], dim=0)
        logits = q[:2 * batch_size].mm(q.T)
        logits = torch.div(logits, self.T)

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # class-averaging
        exp_logits = torch.exp(logits) * logits_mask
        per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(
            2 * batch_size, 2 * batch_size + self.num_classes) - mask
        exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)
        # exp_logits_sum = exp_logits.sum(dim=1, keepdim=True)

        # log_prob = logits - torch.log(exp_logits_sum + 1e-12)
        log_prob = logits - torch.log(exp_logits_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.view(2, batch_size).mean()
        return loss

class TargetedSupConPDANoWeightLoss(nn.Module):  # data augmentation
    def __init__(self, temperature=0.07, num_classes=5):
        super(TargetedSupConPDANoWeightLoss, self).__init__()
        self.T = temperature
        self.num_classes = num_classes

    def forward(self, q, labels, prototypes, prototypes_labels):
        device = (torch.device('cuda')
                  if q.is_cuda
                  else torch.device('cpu'))
        batch_size = labels.shape[0]
        targets = labels.contiguous().view(-1, 1)
        # targets_prototypes = torch.arange(self.num_classes, device=device).view(-1, 1)
        targets = torch.cat([targets.repeat(2, 1), prototypes_labels], dim=0)
        # batch_cls_count = torch.eye(self.num_classes)[targets].sum(dim=0).squeeze()

        mask = torch.eq(targets[:2 * batch_size], targets.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(2 * batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # class-complement
        q = torch.cat([q, prototypes], dim=0)
        logits = q[:2 * batch_size].mm(q.T)
        logits = torch.div(logits, self.T)

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # class-averaging
        exp_logits = torch.exp(logits) * logits_mask
        # per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(
        #     2 * batch_size, 2 * batch_size + self.num_classes) - mask
        # exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)
        exp_logits_sum = exp_logits.sum(dim=1, keepdim=True)

        # log_prob = logits - torch.log(exp_logits_sum + 1e-12)
        log_prob = logits - torch.log(exp_logits_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.view(2, batch_size).mean()
        return loss

class TargetedSupConNoWeightLoss(nn.Module):  # data augmentation
    def __init__(self, temperature=0.07, num_classes=5):
        super(TargetedSupConNoWeightLoss, self).__init__()
        self.T = temperature
        self.num_classes = num_classes

    def forward(self, q, labels, prototypes, prototypes_labels):
        device = (torch.device('cuda')
                  if q.is_cuda
                  else torch.device('cpu'))
        batch_size = labels.shape[0]
        targets = labels.contiguous().view(-1, 1)
        # targets_prototypes = torch.arange(self.num_classes, device=device).view(-1, 1)
        targets = torch.cat([targets, prototypes_labels], dim=0)
        # batch_cls_count = torch.eye(self.num_classes)[targets].sum(dim=0).squeeze()

        mask = torch.eq(targets[: batch_size], targets.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # class-complement
        q = torch.cat([q, prototypes], dim=0)
        logits = q[:batch_size].mm(q.T)
        logits = torch.div(logits, self.T)

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # class-averaging
        exp_logits = torch.exp(logits) * logits_mask
        # per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(
        #     2 * batch_size, 2 * batch_size + self.num_classes) - mask
        # exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)
        exp_logits_sum = exp_logits.sum(dim=1, keepdim=True)

        # log_prob = logits - torch.log(exp_logits_sum + 1e-12)
        log_prob = logits - torch.log(exp_logits_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.mean()
        return loss

class SupConLossV2(nn.Module):  # data augmentation
    def __init__(self, temperature=0.07, num_classes=5):
        super(SupConLossV2, self).__init__()
        self.T = temperature
        self.num_classes = num_classes

    def forward(self, q, labels):
        device = (torch.device('cuda')
                  if q.is_cuda
                  else torch.device('cpu'))
        batch_size = labels.shape[0]
        targets = labels.contiguous().view(-1, 1).repeat(2, 1)

        mask = torch.eq(targets, targets.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(2 * batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # class-complement
        logits = q.mm(q.T)
        logits = torch.div(logits, self.T)

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # class-averaging
        exp_logits = torch.exp(logits) * logits_mask
        # per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(
        #     2 * batch_size, 2 * batch_size + self.num_classes) - mask
        # exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)
        exp_logits_sum = exp_logits.sum(dim=1, keepdim=True)

        # log_prob = logits - torch.log(exp_logits_sum + 1e-12)
        log_prob = logits - torch.log(exp_logits_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.view(2, batch_size).mean()
        return loss
