# encoding: utf-8
'''
@author: lwp, syl
'''

import os
import torch
import numpy as np
import utils.utility as utility
from scipy.spatial.distance import cdist
from utils.functions import cmc, mean_ap

from utils.lr_scheduler import WarmupMultiStepLR
from loss.center_loss import CenterLoss


class Trainer():
    def __init__(self, args, model, loss, center_criterion, loader, ckpt):
        self.args = args
        self.train_loader = loader.train_loader
        self.test_loader = loader.test_loader
        self.query_loader = loader.query_loader
        self.testset = loader.testset
        self.queryset = loader.queryset

        self.ckpt = ckpt
        self.model = model
        self.loss = loss
        self.center_criterion = center_criterion
        self.lr = 0.
        
        # center loss
        if args.center == 'yes':
            self.optimizer, self.optimizer_center = utility.make_optimizer(args, self.model, self.center_criterion)
        else:
            self.optimizer = utility.make_optimizer(args, self.model, self.center_criterion)
        
        # warmup
        if args.warmup == 'yes':
            self.scheduler = WarmupMultiStepLR(self.optimizer, args.decay_list, args.gamma, args.warmup_factor, args.warmup_iters, args.warmup_method)
        else:
            self.scheduler = utility.make_scheduler(args, self.optimizer)
        
        self.device = torch.device('cpu' if args.cpu else 'cuda')

        if args.load != '':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckpt.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckpt.log)*args.test_every): self.scheduler.step()

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]
        if lr != self.lr:
            self.ckpt.write_log('[INFO] Epoch: {}\tLearning rate: {:.2e}'.format(epoch, lr))
            self.lr = lr
        self.loss.start_log()
        self.model.train()

        for batch, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            # center loss
            if self.args.center == 'yes':
                self.optimizer_center.zero_grad()
            
            outputs = self.model(inputs)
            loss = self.loss(outputs, labels)
            
            loss.backward()

            self.optimizer.step()
            # center loss
            if self.args.center == 'yes':
                for param in self.center_criterion.parameters():
                    param.grad.data *= (1. / self.args.center_loss_weight)
                self.optimizer_center.step()

            self.ckpt.write_log('\r[INFO] [{}/{}]\t{}/{}\t{}'.format(
                epoch, self.args.epochs,
                batch + 1, len(self.train_loader),
                self.loss.display_loss(batch)), 
            end='' if batch+1 != len(self.train_loader) else '\n', progress=True)

        self.loss.end_log(len(self.train_loader))

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckpt.write_log('\n[INFO] Test:')
        self.model.eval()

        self.ckpt.add_log(torch.zeros(1, 5))
        qf_tensor = self.extract_feature(self.query_loader)
        gf_tensor = self.extract_feature(self.test_loader)
        
        # no rerank
        '''
        dist = 1-torch.mm(qf_tensor, gf_tensor.t()) # cosine
        dist_np = dist.numpy()
        '''
        
        dist_np = cdist(qf_tensor.numpy(), gf_tensor.numpy(), metric='euclidean')

        m_ap = mean_ap(dist_np, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)
        r = cmc(dist_np, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
                separate_camera_set=False,
                single_gallery_shot=False,
                first_match_break=True)

        self.ckpt.log[-1, 0] = m_ap
        self.ckpt.log[-1, 1] = r[0]
        self.ckpt.log[-1, 2] = r[2]
        self.ckpt.log[-1, 3] = r[4]
        self.ckpt.log[-1, 4] = r[9]
        best = self.ckpt.log.max(0)
        self.ckpt.write_log(
            '[INFO]( ^_^ )  mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f} (Best: {:.4f} @epoch {})'.format(
            m_ap,
            r[0], r[2], r[4], r[9],
            best[0][0],
            (best[1][0] + 1)*self.args.test_every
            )
        )
        
        if not self.args.test_only:
            self.ckpt.save(self, epoch, is_best=((best[1][0] + 1)*self.args.test_every == epoch))

    def fliphor(self, inputs):
        '''
        flip horizontal
        '''
        inv_idx = torch.arange(inputs.size(3)-1,-1,-1).long()  # N x C x H x W
        img_flip = inputs.index_select(3,inv_idx)
        return img_flip
    
    def L2Normalization(self, ff, dim): # add by lwp, 2020.8.20
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        return ff

    def extract_feature(self, loader): # modified by lwp, 2020.8.20
        features = torch.FloatTensor()
        for (inputs, labels) in loader:
            ff = torch.FloatTensor(inputs.size(0), 4096).zero_().cuda()
            for i in range(2):
                if i==1:
                    inputs = self.fliphor(inputs)
                input_img = inputs.to(self.device)
                outputs = self.model(input_img)
                ff += outputs[0]

            ff = self.L2Normalization(ff, dim=1) 
            features = torch.cat((features, ff.data.cpu().float()), 0)
        return features

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

