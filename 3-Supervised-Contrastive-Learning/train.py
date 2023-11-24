from __future__ import print_function

import glob
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score
from torch import nn


import os
import argparse
import time
from PIL import Image
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from timm.scheduler import CosineLRScheduler
import tensorboard_logger as tb_logger
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models as torchvision_models
from torch.utils.data import DataLoader, Dataset

from dataset import CellDatasetV2 as CellDataset, CellDataset as ValidCellDataset
from supcon_util import AverageMeter, GaussianBlur
from scl_model import SupConVitGAClassifier as Classifier
from supcon_loss import TargetedSupConPDALoss, TargetedSupConPDANoWeightLoss

class Evaluator(object):
    def __init__(self, n_class):
        self.metrics = ConfusionMatrix(n_class)

    def get_scores(self):
        acc = self.metrics.get_scores()

        return acc

    def reset_metrics(self):
        self.metrics.reset()

    def plot_cm(self):
        self.metrics.plotcm()

    def eval_test(self, sample, model):
        node_feat, labels = prepareCellfeatureLabel(sample['image'], sample['label'])
        with torch.no_grad():
            output = model.forward(node_feat)
        return output,labels


def cell_collate(batch):
    image = [b['image'] for b in batch]  # w, h
    label = [b['label'] for b in batch]
    id = [b['id'] for b in batch]
    return {'image': image, 'label': label, 'id': id}

def prepareCellfeatureLabel(batch_graph, batch_label):
    batch_size = len(batch_graph)
    labels = torch.LongTensor(batch_size)
    max_node_num = 0

    for i in range(batch_size):
        labels[i] = batch_label[i]
        max_node_num = max(max_node_num, batch_graph[i].shape[0])

    # batch_node_feat = torch.zeros(batch_size, max_node_num, 1536)
    batch_node_feat = torch.zeros(batch_size, max_node_num, 2048)
    # batch_node_feat = torch.zeros(batch_size, max_node_num, 1280)

    for i in range(batch_size):
        cur_node_num = batch_graph[i].shape[0]
        # node attribute feature
        tmp_node_fea = batch_graph[i]
        batch_node_feat[i, 0:cur_node_num] = tmp_node_fea

    node_feat = batch_node_feat.cuda()
    labels = labels.cuda()

    return node_feat, labels


class DataAugmentationDINO(object):
    def __init__(self):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.Resize(224),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])

        self.unchange = transforms.Compose([
            transforms.Resize(224),
            normalize,
        ])

    def __call__(self, image):
        probability = 0.5
        random_number = random.random()

        if random_number < probability:
            q = self.global_transfo1(image)
        else:
            q = self.unchange(image)

        random_number = random.random()
        if random_number < probability:
            k = self.global_transfo1(image)
        else:
            k = self.unchange(image)
        return [q, k]


class BagDataset(Dataset):
    def __init__(self, csv_file, transform):
        self.files_list = glob.glob(os.path.join(csv_file, '*.jpg'))
        self.transform = transform

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        img = Image.open(self.files_list[idx])
        return self.transform(img)


def load_pretrain():
    model = torchvision_models.__dict__['resnet50']()
    # pretrained_weights = '/home/nas2/path/wanglang_22/deep-learning/ssl/dino/checkpoint/c1/checkpoint.pth'
    pretrained_weights = '/home/nas2/path/wanglang_22/deep-learning/ssl/dino/checkpoint/c2/checkpoint.pth'
    checkpoint_key = 'teacher'
    model.fc = nn.Identity()
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        state_dict = state_dict[checkpoint_key]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.cuda()
    for param in model.parameters():
        param.requires_grad = False
    return model


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=1000,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=10,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    parser.add_argument('--train_set', type=str,
                        default='/home/nas2/path/wanglang_22/deep-learning/cell/p_train.csv',
                        help='train')
    parser.add_argument('--val_set', type=str,
                        default='/home/nas2/path/wanglang_22/deep-learning/cell/p_valid.csv',
                        help='validation')
    parser.add_argument('--model_path', type=str,
                        default='/home/nas2/path/wanglang_22/deep-learning/cervical/checkpoint/cell-tmi/',
                        help='path to trained model')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                        help='learning rate')
    opt = parser.parse_args()

    opt.tb_folder = os.path.join(opt.model_path, 'tensorboard')
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder, exist_ok=True)
    return opt


class MoCo(nn.Module):
    """
        Build a MoCo model with: a query encoder, a key encoder, and a queue
        https://arxiv.org/abs/1911.05722
        """

    def __init__(self, dim=128, tr=1):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()
        self.n_cls = 5
        self.tr = tr
        optimal_target = np.load('optimal_{}_{}.npy'.format(self.n_cls, dim))
        optimal_target_order = np.arange(self.n_cls)

        target_repeat = tr * np.ones(self.n_cls)

        optimal_target = torch.Tensor(optimal_target).float()
        target_repeat = torch.Tensor(target_repeat).long()
        optimal_target = torch.cat(
            [optimal_target[i:i + 1, :].repeat(target_repeat[i], 1) for i in range(len(target_repeat))], dim=0)

        target_labels = torch.cat(
            [torch.Tensor([optimal_target_order[i]]).repeat(target_repeat[i]) for i in range(len(target_repeat))],
            dim=0).long().unsqueeze(-1)

        self.register_buffer("optimal_target", optimal_target)
        self.register_buffer("target_labels", target_labels)

        # create the queue
        self.register_buffer("class_centroid", torch.randn(self.n_cls, dim))
        self.class_centroid = nn.functional.normalize(self.class_centroid, dim=1)

    def forward(self, im_q, im_labels):
        """
        Input:
            im_q: a batch of query images
        Output:
            targets, assignment
        """
        # compute the optimal matching that minimize moving distance between memory bank anchors and targets
        with torch.no_grad():
            features_all = im_q.detach()
            # update memory bank class centroids
            for one_label in torch.unique(im_labels):
                class_centroid_batch = F.normalize(
                    torch.mean(features_all[im_labels[:, 0].eq(one_label), :], dim=0), dim=0)
                self.class_centroid[one_label] = 0.9 * self.class_centroid[one_label] + 0.1 * class_centroid_batch
                self.class_centroid[one_label] = F.normalize(self.class_centroid[one_label], dim=0)

            centroid_target_dist = torch.einsum('nc,ck->nk', [self.class_centroid, self.optimal_target.T])
            centroid_target_dist = centroid_target_dist.detach().cpu().numpy()

            row_ind, col_ind = linear_sum_assignment(-centroid_target_dist)

            for one_label, one_idx in zip(row_ind, col_ind):
                self.target_labels[one_idx:0] = one_label

        return self.optimal_target, self.target_labels.clone().detach()


def train(train_loader, model, optimizer, criterion, epoch, opt, prototypes, r50, transform):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()

    # accumulation_steps = 192
    # accumulation_steps = 128
    accumulation_steps = 64
    outputs_q = torch.Tensor().cuda()
    outputs_k = torch.Tensor().cuda()
    targets = torch.Tensor().to(torch.long).cuda()
    for idx, (samples, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # data augmentation
        bag = BagDataset(samples[0], transform=transform)
        dl = DataLoader(bag, batch_size=32, shuffle=False, num_workers=8, drop_last=False)
        qt = torch.Tensor().cuda()
        kt = torch.Tensor().cuda()
        with torch.no_grad():
            for iteration, batch in enumerate(
                    dl):  # batch: list size of 2, each item size of batch_size * 3 * 224 224 , q and k tensor
                # print(type(batch))
                q, k = batch[0].cuda(), batch[1].cuda()
                # print(type(q))
                # print(len(q))
                bs = q.shape[0]
                inputs = torch.cat([q, k], dim=0)
                # inputs = inputs.cuda()
                inputs = r50(inputs)
                q, k = torch.split(inputs, [bs, bs], dim=0)
                qt = torch.cat((qt, q), dim=0)
                kt = torch.cat([kt, k], dim=0)
        qt = qt.unsqueeze(0)
        kt = kt.unsqueeze(0)
        qk = torch.cat([qt, kt], dim=0)
        qk = model(qk)
        qt, kt = torch.split(qk, [1, 1], dim=0)
        outputs_q = torch.cat((outputs_q, qt), dim=0)
        outputs_k = torch.cat([outputs_k, kt], dim=0)
        labels = labels.cuda()
        targets = torch.cat((targets, labels.contiguous().view(-1, 1)), dim=0)
        if (idx + 1) % accumulation_steps == 0:
            optimizer.zero_grad()
            outputs_q = torch.cat([outputs_q, outputs_k], dim=0)
            loss = criterion(outputs_q, targets, prototypes)
            loss.backward()
            optimizer.step()
            outputs_q = torch.Tensor().cuda()
            outputs_k = torch.Tensor().cuda()
            targets = torch.Tensor().to(torch.long).cuda()
            # update metric
            losses.update(loss.item(), accumulation_steps)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # print info
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            sys.stdout.flush()
    if (idx + 1) % accumulation_steps != 0:
        optimizer.zero_grad()
        outputs_q = torch.cat([outputs_q, outputs_k], dim=0)
        loss = criterion(outputs_q, targets, prototypes)
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), accumulation_steps)
        batch_time.update(time.time() - end)
        end = time.time()
        print('Train: [{0}][{1}/{2}]\t'
              'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
            epoch, idx + 1, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses))
        sys.stdout.flush()
    return losses.avg


def tsne_analysis(model, epoch, path, classifier, opt):
    valid_set = '/home/nas2/path/wanglang_22/deep-learning/cell/p_valid.csv'
    dataset_val = ValidCellDataset(valid_set)
    val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=1, num_workers=1,
                                             collate_fn=cell_collate, shuffle=False, pin_memory=True)
    evaluator = Evaluator(5)

    with torch.no_grad():
        model.eval()
        print("evaluating...")

        output_all = torch.randn((0, 5))
        label_all = []

        for i_batch, sample_batched in enumerate(val_loader):
            output, labels = evaluator.eval_test(sample_batched, model)
            output = classifier(output)
            preds = output.data.max(1)[1]

            evaluator.metrics.update(labels, preds)

            output_all = torch.cat([output_all, output.data.cpu()], dim=0)
            label_all.append(labels.item())

        # print('[%d/%d] val agg acc: %.3f' % (494, 494, evaluator.get_scores()))
        evaluator.plot_cm()

        # auc
        output_all = F.softmax(output_all, dim=1)
        output_all = np.array(output_all)
        label_all = np.array(label_all)
        auc = roc_auc_score(label_all, output_all, multi_class='ovr')

        val_acc = evaluator.get_scores()
        if val_acc > opt.best_pred:
            opt.best_pred = val_acc
        if auc > opt.best_auc:
            opt.best_auc = auc

        log = ""
        log = log + 'epoch [{}/{}] ------ acc: val = {:.4f}, auc = {:.4f}'.format(epoch,
                                                                                  200,
                                                                                  evaluator.get_scores(),
                                                                                  auc) + "\n"
        log += "================================\n"
        print(log)
    print('best acc: {:.4f}, best auc: {:.4f}'.format(opt.best_pred, opt.best_auc))

    # model.eval()
    # feats_list = []
    # labels_list = []
    # with torch.no_grad():
    #     for idx, sample in enumerate(val_loader):
    #         node_feat, labels = prepareCellfeatureLabel(sample['image'], sample['label'])
    #         feats = model.forward(node_feat).flatten().cpu().numpy()
    #         feats_list.append(feats)
    #         labels_list.append(labels.flatten().item())
    # feats_list = np.array(feats_list)
    # labels_list = np.array(labels_list)
    # tsne = TSNE(n_components=2, random_state=42)
    # embedded_features = tsne.fit_transform(feats_list)
    # plt.figure(figsize=(10, 8))
    # plt.scatter(embedded_features[:, 0], embedded_features[:, 1], c=labels_list, cmap=plt.cm.get_cmap("jet", 5))
    # plt.colorbar(ticks=range(5))
    # plt.title('t-SNE Visualization')
    # plt.savefig(os.path.join(path, '{}.png'.format(epoch)), dpi=300)
    # plt.close()


def train_with_cache(train_loader, model, optimizer, criterion, epoch, targeted, classifier, classifier_criterion):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()

    # accumulation_steps = 192
    # accumulation_steps = 128
    # accumulation_steps = 32
    # accumulation_steps = 64
    accumulation_steps = 64
    outputs_q = torch.Tensor().cuda()
    outputs_k = torch.Tensor().cuda()
    targets = torch.Tensor().to(torch.long).cuda()
    # for classification
    logit = torch.Tensor().cuda()
    for idx, (samples, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # data augmentation with cache
        q_path = os.path.join(samples[0], str(epoch - 1), 'q.csv')
        k_path = os.path.join(samples[0], str(epoch - 1), 'k.csv')
        qt = torch.tensor(pd.read_csv(q_path).values).float().cuda()
        kt = torch.tensor(pd.read_csv(k_path).values).float().cuda()
        qt = qt.unsqueeze(0)
        kt = kt.unsqueeze(0)
        qk = torch.cat([qt, kt], dim=0)
        qk, m = model(qk)
        qt, kt = torch.split(qk, [1, 1], dim=0)
        outputs_q = torch.cat((outputs_q, qt), dim=0)
        outputs_k = torch.cat([outputs_k, kt], dim=0)
        labels = labels.cuda()
        targets = torch.cat((targets, labels.contiguous().view(-1, 1)), dim=0)
        logit = torch.cat([logit, classifier(m).view(1, -1)], dim=0)
        if (idx + 1) % accumulation_steps == 0:
            optimizer.zero_grad()
            prototypes, prototypes_targets = targeted(outputs_q, targets)
            outputs_q = torch.cat([outputs_q, outputs_k], dim=0)
            loss = criterion(outputs_q, targets, prototypes, prototypes_targets)
            cls_loss = classifier_criterion(logit, targets.squeeze())
            loss.backward()
            cls_loss.backward()
            optimizer.step()
            outputs_q = torch.Tensor().cuda()
            outputs_k = torch.Tensor().cuda()
            targets = torch.Tensor().to(torch.long).cuda()
            logit = torch.Tensor().cuda()
            # update metric
            losses.update(loss.item(), accumulation_steps)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # print info
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            sys.stdout.flush()
    if (idx + 1) % accumulation_steps != 0:
        optimizer.zero_grad()
        prototypes, prototypes_targets = targeted(outputs_q, targets)
        outputs_q = torch.cat([outputs_q, outputs_k], dim=0)
        loss = criterion(outputs_q, targets, prototypes, prototypes_targets)
        cls_loss = classifier_criterion(logit, targets.squeeze())
        loss.backward()
        cls_loss.backward()
        optimizer.step()
        losses.update(loss.item(), accumulation_steps)
        batch_time.update(time.time() - end)
        end = time.time()
        print('Train: [{0}][{1}/{2}]\t'
              'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
            epoch, idx + 1, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses))
        sys.stdout.flush()
    return losses.avg


def main():
    opt = parse_option()
    opt.best_pred = 0.0
    opt.best_auc = 0.0
    tsne_path = os.path.join(opt.model_path, 'tsne')
    os.makedirs(tsne_path, exist_ok=True)
    dataset_train = CellDataset(opt.train_set)
    dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=1, num_workers=1,
                                                   shuffle=True, pin_memory=True,
                                                   drop_last=False)
    model = Classifier()
    # criterion = TargetedSupConPDANoWeightLoss(temperature=0.1)
    criterion = TargetedSupConPDANoWeightLoss()
    # criterion=TargetedSupConPDALoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    # prototypes
    targeted = MoCo()
    if torch.cuda.is_available():
        targeted = targeted.cuda()
    # data augmentation
    # r50 = load_pretrain()
    # transform = DataAugmentationDINO()

    # classifier
    classifier = nn.Linear(768, 5)
    classifier_criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        classifier = classifier.cuda()
        classifier_criterion = classifier_criterion.cuda()

    param_groups = [
        {'params': model.parameters(), 'lr': 0.0002},
        {'params': classifier.parameters(), 'lr': 0.0002},
    ]
    # param_groups = [
    #     {'params': model.parameters(), 'lr': 0.0003},
    #     {'params': classifier.parameters(), 'lr': 0.0003},
    # ]
    optimizer = torch.optim.Adam(param_groups, weight_decay=5e-4)  # best:5e-4, 4e-3
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, 0.000005)
    # scheduler = CosineLRScheduler(
    #     optimizer,
    #     t_initial=opt.epochs,
    #     lr_min=0.000005,
    #     warmup_lr_init=1e-5,
    #     warmup_t=5,
    # )
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=opt.epochs,
        lr_min=0.000005
    )
    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    for epoch in range(1, opt.epochs + 1):
        time1 = time.time()
        loss = train_with_cache(dataloader_train, model, optimizer, criterion, epoch, targeted, classifier,
                                classifier_criterion)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        scheduler.step(epoch)

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        print("saving model...")
        torch.save(model.state_dict(), os.path.join(opt.model_path, str(epoch) + "_" + str(loss) + ".pth"))

        # tsne analysis
        tsne_analysis(model, epoch, tsne_path, classifier, opt)


if __name__ == '__main__':
    main()
