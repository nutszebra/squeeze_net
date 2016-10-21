import six
import numpy as np
import functools
import chainer.links as L
import chainer.functions as F
from collections import defaultdict
import nutszebra_chainer


class BN_ReLU_Conv(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        super(BN_ReLU_Conv, self).__init__(
            conv=L.Convolution2D(in_channel, out_channel, filter_size, stride, pad),
            bn=L.BatchNormalization(in_channel),
        )

    def weight_initialization(self):
        self.conv.W.data = self.weight_relu_initialization(self.conv)
        self.conv.b.data = self.bias_initialization(self.conv, constant=0)

    def __call__(self, x, train=False):
        return self.conv(F.relu(self.bn(x, test=not train)))

    def count_parameters(self):
        return functools.reduce(lambda a, b: a * b, self.conv.W.data.shape)


class FireModule(nutszebra_chainer.Model):

    def __init__(self, in_size, s1x1, e1x1, e3x3):
        super(FireModule, self).__init__()
        modules = []
        modules.append(('bn_relu_conv_s_1x1', BN_ReLU_Conv(in_size, s1x1, filter_size=(1, 1), stride=(1, 1), pad=(0, 0))))
        modules.append(('bn_relu_conv_e_1x1', BN_ReLU_Conv(s1x1, e1x1, filter_size=(1, 1), stride=(1, 1), pad=(0, 0))))
        modules.append(('bn_relu_conv_e_3x3', BN_ReLU_Conv(s1x1, e3x3, filter_size=(3, 3), stride=(1, 1), pad=(1, 1))))
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules

    def weight_initialization(self):
        self.bn_relu_conv_s_1x1.weight_initialization()
        self.bn_relu_conv_e_1x1.weight_initialization()
        self.bn_relu_conv_e_3x3.weight_initialization()

    def __call__(self, x, train=False):
        h = self.bn_relu_conv_s_1x1(x, train=train)
        h1 = self.bn_relu_conv_e_1x1(h, train=train)
        h2 = self.bn_relu_conv_e_3x3(h, train=train)
        return F.concat((h1, h2), axis=1)

    def count_parameters(self):
        count = 0
        count += self.bn_relu_conv_s_1x1.count_parameters()
        count += self.bn_relu_conv_e_1x1.count_parameters()
        count += self.bn_relu_conv_e_3x3.count_parameters()
        return count


class SqueezeNet(nutszebra_chainer.Model):

    def __init__(self, category_num):
        super(SqueezeNet, self).__init__()
        modules = []
        modules = []
        modules += [('conv1', L.Convolution2D(3, 96, (7, 7), (2, 2), (2, 2)))]
        # fire module(in_size, s1x1, e1x1, e3x3)
        modules += [('fire2', FireModule(96, 16, 64, 64))]
        modules += [('fire3', FireModule(128, 16, 64, 64))]
        modules += [('fire4', FireModule(128, 32, 128, 128))]
        modules += [('fire5', FireModule(256, 32, 128, 128))]
        modules += [('fire6', FireModule(256, 48, 192, 192))]
        modules += [('fire7', FireModule(384, 48, 192, 192))]
        modules += [('fire8', FireModule(384, 64, 256, 256))]
        modules += [('fire9', FireModule(512, 64, 256, 256))]
        modules += [('bn_relu_conv10', BN_ReLU_Conv(512, category_num, (1, 1), (1, 1), (0, 0)))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.name = 'squeeze_res_net'

    def weight_initialization(self):
        self.conv1.W.data = self.weight_relu_initialization(self.conv1)
        self.conv1.b.data = self.bias_initialization(self.conv1, constant=0)
        # *****fire modules*****
        for i in six.moves.range(2, 10):
            self['fire{}'.format(i)].weight_initialization()
        self.bn_relu_conv10.weight_initialization()

    def __call__(self, x, train=True):
        h = self.conv1(x)
        h = F.max_pooling_2d(h, ksize=(3, 3), stride=(2, 2), pad=(1, 1))
        h = self.fire2(h, train=train)
        h = self.fire3(h, train=train) + h
        h = self.fire4(h, train=train)
        h = F.max_pooling_2d(h, ksize=(3, 3), stride=(2, 2), pad=(1, 1))
        h = self.fire5(h, train=train) + h
        h = self.fire6(h, train=train)
        h = self.fire7(h, train=train) + h
        h = self.fire8(h, train=train)
        h = F.max_pooling_2d(h, ksize=(3, 3), stride=(2, 2), pad=(1, 1))
        h = self.fire9(h, train=train) + h
        h = self.bn_relu_conv10(h, train=train)
        num, categories, y, x = h.data.shape
        # global average pooling
        h = F.reshape(F.average_pooling_2d(h, (y, x)), (num, categories))
        return h

    def calc_loss(self, y, t):
        loss = F.softmax_cross_entropy(y, t)
        return loss

    def accuracy(self, y, t, xp=np):
        y.to_cpu()
        t.to_cpu()
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == True)[0]
        accuracy = defaultdict(int)
        for i in indices:
            accuracy[t.data[i]] += 1
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == False)[0]
        false_accuracy = defaultdict(int)
        false_y = np.argmax(y.data, axis=1)
        for i in indices:
            false_accuracy[(t.data[i], false_y[i])] += 1
        return accuracy, false_accuracy
