import chainer
from chainer import optimizers
import nutszebra_basic_print
from nutszebra_utility import Utility as utility


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
