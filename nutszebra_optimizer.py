import chainer
from chainer import optimizers
import nutszebra_basic_print
from nutszebra_utility import Utility as utility
import numpy as np


class Optimizer(object):

    def __init__(self, model=None):
        self.model = model
        self.optimizer = None

    def __call__(self, i):
        pass

    def update(self):
        self.optimizer.update()


class ILSVRC(Optimizer):

    def __init__(self, model=None, lr_file='./lr.txt', momentum=0.9, weight_decay=1.0e-4):
        super(ILSVRC, self).__init__(model)
        lr = self.load_lr(lr_file)
        print('initial lr: {}'.format(lr))
        optimizer = optimizers.MomentumSGD(lr, momentum)
        wd = chainer.optimizer.WeightDecay(weight_decay)
        optimizer.setup(self.model)
        optimizer.add_hook(wd)
        self.optimizer = optimizer
        self.lr_file = lr_file
        self.count = 0

    def __call__(self, i):
        pass

    def update(self):
        self.optimizer.update()
        # check lr
        self.count += 1
        if self.count >= 10:
            self.count = 0
            lr = self.load_lr(self.lr_file)
            if not self.optimizer.lr == lr:
                print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
                self.optimizer.lr = lr

    def load_lr(self, path):
        return float(utility.load_text(path)[0])


class OptimizerCosineAnnealing(Optimizer):

    def __init__(self, model=None, eta_max=0.1, eta_min=0.1 * 10 ** -3, total_epoch=200, momentum=0.9, weight_decay=1.0e-4, start_epoch=0):
        super(OptimizerCosineAnnealing, self).__init__(model)
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.total_epoch = total_epoch
        lr = OptimizerCosineAnnealing.calc_lr(eta_min, eta_max, start_epoch, total_epoch)
        print('initial learing rate: {}'.format(lr))
        optimizer = optimizers.MomentumSGD(lr, momentum)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        optimizer.setup(self.model)
        optimizer.add_hook(weight_decay)
        self.optimizer = optimizer

    @staticmethod
    def calc_lr(eta_min, eta_max, i, total_epoch):
        return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * float(i) / total_epoch))

    def __call__(self, i):
        new_lr = OptimizerCosineAnnealing.calc_lr(self.eta_min, self.eta_max, i, self.total_epoch)
        old_lr = self.optimizer.lr
        print('lr is changed: {} -> {}'.format(old_lr, new_lr))
        self.optimizer.lr = new_lr
