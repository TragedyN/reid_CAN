# encoding: utf-8
'''
@author: lwp, syl
'''

import data
import loss
import torch
import model
from trainer import Trainer

from option import args
import utils.utility as utility

ckpt = utility.checkpoint(args)

loader = data.Data(args)
model   = model.Model(args, ckpt)

loss = loss.Loss(args, ckpt) if not args.test_only else None

if args.center == 'yes':
    center_criterion = loss.loss_module[-1] if not args.test_only else None
    trainer = Trainer(args, model, loss, center_criterion, loader, ckpt)
else:
    center_criterion = -1
    trainer = Trainer(args, model, loss, center_criterion, loader, ckpt)

n = 0
while not trainer.terminate():
	n += 1
	trainer.train()
	if args.test_every!=0 and n%args.test_every==0:
		trainer.test()


