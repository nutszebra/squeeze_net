# """model definition."""
import numpy as np
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import cuda
from chainer import serializers


class BN_Relu_Conv(chainer.Chain):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        """Init."""
        initializer = chainer.initializers.HeNormal()
        super(BN_Relu_Conv, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(in_channel, out_channel, filter_size, stride, pad, initialW=initializer)
            self.bn = L.BatchNormalization(in_channel)

    def __call__(self, x):
        """Call."""
        h = F.relu(self.bn(x))
        h = self.conv(h)
        return h


class FireModule(chainer.Chain):

    def __init__(self, in_size, s1, e1, e3):
        """Init."""
        super(FireModule, self).__init__()
        with self.init_scope():
            self.conv1 = BN_Relu_Conv(in_size, s1, filter_size=(1, 1), stride=(1, 1), pad=(0, 0))
            self.conv2 = BN_Relu_Conv(s1, e1, filter_size=(1, 1), stride=(1, 1), pad=(0, 0))
            self.conv3 = BN_Relu_Conv(s1, e3, filter_size=(3, 3), stride=(1, 1), pad=(1, 1))

    def __call__(self, x):
        """Call."""
        h = self.conv1(x)
        h1 = self.conv2(h)
        h2 = self.conv3(h)
        h_expand = F.concat([h1, h2], axis=1)
        return h_expand


class Squeeze(chainer.Chain):

    def __init__(self, category_num=10):
        """Init."""
        initializer = chainer.initializers.HeNormal()
        super(Squeeze, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 96, (7, 7), (2, 2), (2, 2),
                                         initialW=initializer)
            self.fire2 = FireModule(96, 16, 64, 64)
            self.fire3 = FireModule(128, 16, 64, 64)
            self.fire4 = FireModule(128, 32, 128, 128)
            self.fire5 = FireModule(256, 32, 128, 128)
            self.fire6 = FireModule(256, 48, 192, 192)
            self.fire7 = FireModule(384, 48, 192, 192)
            self.fire8 = FireModule(384, 64, 256, 256)
            self.fire9 = FireModule(512, 64, 256, 256)
            self.conv10 = BN_Relu_Conv(512, category_num, (1, 1), (1, 1), (0, 0))

    def __call__(self, x):
        """Call."""
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 3, stride=2, pad=1)
        h = self.fire2(h)
        h = self.fire3(h) + h
        h = self.fire4(h)
        h = F.max_pooling_2d(h, 3, stride=2, pad=1)
        h = self.fire5(h) + h
        h = self.fire6(h)
        h = self.fire7(h) + h
        h = self.fire8(h)
        h = F.max_pooling_2d(h, 3, stride=2, pad=1)
        h = self.fire9(h)
        h = F.dropout(h, ratio=0.5)
        h = self.conv10(h)
        num, categories, y, x = h.data.shape
        h = F.reshape(F.average_pooling_2d(h, (y, x)), (num, categories))
        return h

    def weight_initialization(self):
            pass

    def check_gpu(self, gpu):
        if gpu >= 0:
            cuda.get_device(gpu).use()
            self.to_gpu(gpu)
            return True
        return False

    @staticmethod
    def _check_cupy():
        try:
            cuda.check_cuda_available()
            return cuda.cupy
        # if gpu is not available, RuntimeError arises
        except RuntimeError:
            return np

    def prepare_input(self, X, dtype=np.float32, volatile=False, xp=None, gpu=None):
        if gpu is not None:
            inp = np.asarray(X, dtype=dtype)
            inp = chainer.Variable(inp, volatile=volatile)
            inp.to_gpu(gpu)
            return inp
        if xp is None:
            if self.model_is_cpu_mode():
                inp = np.asarray(X, dtype=dtype)
            else:
                inp = self.nz_xp.asarray(X, dtype=dtype)
        else:
            inp = xp.asarray(X, dtype=dtype)
        return chainer.Variable(inp, volatile=volatile)

    def save_model(self, path='', gpu=0):
        # if gpu_flag is True, switch the model to gpu mode at last
        gpu_flag = False
        # if gpu mode, switch the model to cpu mode temporarily
        if self.model_is_cpu_mode() is False:
            self.to_cpu()
            gpu_flag = True
        # if path is ''
        if path == '':
            path = str(self.save_model_epoch) + '.model'
        self.nz_save_model_epoch += 1
        # increment self.nz_save_model_epoch
        serializers.save_npz(path, self)
        # if gpu_flag is True, switch the model to gpu mode at last
        if gpu_flag:
            self.to_gpu(gpu)
        return True
