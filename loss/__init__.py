import os
import numpy as np
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from loss.triplet_loss import TripletLoss, TripletHardLoss
from loss.center_loss import CenterLoss



class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckpt):
        super(Loss, self).__init__()
        print('[INFO] Making loss...')
        
        self.nGPU = args.nGPU
        self.args = args
        self.loss = []
        self.loss_module = nn.ModuleList()

        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'CrossEntropy_Loss':
                loss_function = nn.CrossEntropyLoss()
            elif loss_type == 'Triplet_Loss':
                loss_function = TripletLoss(args.margin)
            elif loss_type == 'TripletHard_Loss':
                loss_function = TripletHardLoss(args.margin)
            elif loss_type == 'Center_Loss':
                loss_function = CenterLoss(num_classes=args.num_classes, feat_dim=256, use_gpu=True) 

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function
                })
            

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.4f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()

        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        
        if args.load != '': self.load(ckpt.dir, cpu=args.cpu)
        if not args.cpu and args.nGPU > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.nGPU)
            )

    def forward(self, outputs, labels):
        losses = []
        for i, l in enumerate(self.loss):
            if self.args.model == 'CAN' and l['type'] == 'Triplet_Loss':
                loss = [l['function'](output, labels) for output in outputs[1:8]]
                loss = sum(loss) / len(loss)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item() 
            elif self.args.model == 'CAN' and l['type'] == 'TripletHard_Loss':
                loss = [l['function'](output, labels) for output in outputs[1:8]]
                loss = sum(loss) / len(loss)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            elif self.args.model == 'CAN' and l['type'] == 'CrossEntropy_Loss':
                loss = [l['function'](output, labels) for output in outputs[8:]]
                loss = sum(loss) / len(loss)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            elif self.args.model == 'CAN' and l['type'] == 'Center_Loss':
                loss = []
                for output in outputs[1:8]:
                    l['function'].set_feat_dim(output.shape[1])
                    loss_c = l['function'](output, labels)
                    loss.append(loss_c) 
                loss = sum(loss) / len(loss)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            else:
                pass
        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()
        
        return loss_sum

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, batches):
        self.log[-1].div_(batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('{}/loss_{}.png'.format(apath, l['type']))
            plt.close(fig)

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def get_loss_module(self):
        if self.nGPU == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.loss_module:
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()

