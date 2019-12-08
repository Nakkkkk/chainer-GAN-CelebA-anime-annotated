#!/usr/bin/python
# coding: UTF-8

# original code URL https://github.com/xkumiyu/chainer-GAN-CelebA
# revised by Nakkkkk(https://github.com/Nakkkkk)

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L


def add_noise(h, sigma=0.2):
    xp = cuda.get_array_module(h.data)
    if chainer.config.train:
        return h + sigma * xp.random.randn(*h.shape)
    else:
        return h

# Minibatch_Discriminationによるモード崩壊の防止（http://musyoku.github.io/2016/12/23/Improved-Techniques-for-Training-GANs/）
class Minibatch_Discrimination(chainer.Chain):
    """
    Minibatch Discrimination Layer
    Parameters
    ---------------------
    B: int
        number of rows of M
    C: int
        number of columns of M
    wscale: float
        std of normal initializer
    """

    def __init__(self, B, C, wscale):
        super(Minibatch_Discrimination, self).__init__()
        self.b = B
        self.c = C
        with self.init_scope():
            # initialozer to W
            w = chainer.initializers.Normal(wscale)

            # register Parameters
            self.t = L.Linear(in_size=None,
                              out_size=B*C,
                              initialW=w,
                              nobias=True)  # bias is required ?

    def __call__(self, x):
        """
        Calucurate Minibatch Discrimination using broardcast.
        Parameters
        ---------------
        x: Variable
           input vector shape is (N, num_units)
        """
        batch_size = x.shape[0]
        xp = x.xp
        activation = self.t(x)
        m = F.reshape(activation, (-1, self.b, self.c))
        m = F.expand_dims(m, 3)
        m_T = F.transpose(m, (3, 1, 2, 0))
        m, m_T = F.broadcast(m, m_T)
        l1_norm = F.sum(F.absolute(m-m_T), axis=2)

        # eraser to erase l1 norm with themselves
        eraser = F.expand_dims(xp.eye(batch_size, dtype="f"), 1)
        eraser = F.broadcast_to(eraser, (batch_size, self.b, batch_size))

        o_X = F.sum(F.exp(-(l1_norm + 1e6 * eraser)), axis=2)
        # concatunate along channels or units
        return F.concat((x, o_X), axis=1)


class Discriminator(chainer.Chain):

    def __init__(self, wscale=0.02, unrolling_steps=5):
        self.b, self.c = 32, 8
        w = chainer.initializers.Normal(wscale)
        self.unrolling_steps = unrolling_steps
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.c0_0 = L.Convolution2D(3, 64, 3, stride=2, pad=1, initialW=w)
            self.c0_1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, initialW=w)
            self.c1_0 = L.Convolution2D(128, 128, 3, stride=1, pad=1, initialW=w)
            self.c1_1 = L.Convolution2D(128, 256, 4, stride=2, pad=1, initialW=w)
            self.c2_0 = L.Convolution2D(256, 256, 3, stride=1, pad=1, initialW=w)
            self.c2_1 = L.Convolution2D(256, 512, 4, stride=2, pad=1, initialW=w)
            #self.c3_0 = L.Convolution2D(512, 512, 3, stride=1, pad=1, initialW=w)
            self.l4_0 = L.Linear(4 * 4 * 512, 128, initialW=w)
            self.md1 = Minibatch_Discrimination(
                B=self.b, C=self.c, wscale=wscale)
            #self.l4 = L.Linear(4 * 4 * 512, 1, initialW=w)
            self.l4 = L.Linear(None, 12, initialW=w)
            self.bn0_1 = L.BatchNormalization(128, use_gamma=False)
            self.bn1_0 = L.BatchNormalization(128, use_gamma=False)
            self.bn1_1 = L.BatchNormalization(256, use_gamma=False)
            self.bn2_0 = L.BatchNormalization(256, use_gamma=False)
            self.bn2_1 = L.BatchNormalization(512, use_gamma=False)
            self.bn3_0 = L.BatchNormalization(512, use_gamma=False)

    def cache_discriminator_weights(self):
        self.cached_weights = {}
        for name, param in self.namedparams():
            with cuda.get_device(param.data):
                xp = cuda.get_array_module(param.data)
                self.cached_weights[name] = xp.copy(param.data)

    def restore_discriminator_weights(self):
        for name, param in self.namedparams():
            with cuda.get_device(param.data):
                if name not in self.cached_weights:
                    raise Exception()
                param.data = self.cached_weights[name]

    def __call__(self, x):
        h = add_noise(x)
        h = F.leaky_relu(add_noise(self.c0_0(h)))
        h = F.leaky_relu(add_noise(self.bn0_1(self.c0_1(h))))
        h = F.leaky_relu(add_noise(self.bn1_0(self.c1_0(h))))
        h = F.leaky_relu(add_noise(self.bn1_1(self.c1_1(h))))
        h = F.leaky_relu(add_noise(self.bn2_0(self.c2_0(h))))
        h = F.leaky_relu(add_noise(self.bn2_1(self.c2_1(h))))
        #h = F.leaky_relu(add_noise(self.bn3_0(self.c3_0(h))))
        h = self.l4_0(h)
        h = self.md1(h)
        h = self.l4(h)
        return h


class Encoder(chainer.Chain):

    def __init__(self, wscale=0.02):
        w = chainer.initializers.Normal(wscale)
        super(Encoder, self).__init__()
        with self.init_scope():
            self.c0_0 = L.Convolution2D(3, 64, 3, stride=2, pad=1, initialW=w)
            self.c0_1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, initialW=w)
            self.c1_0 = L.Convolution2D(128, 128, 3, stride=1, pad=1, initialW=w)
            self.c1_1 = L.Convolution2D(128, 256, 4, stride=2, pad=1, initialW=w)
            self.c2_0 = L.Convolution2D(256, 256, 3, stride=1, pad=1, initialW=w)
            self.c2_1 = L.Convolution2D(256, 512, 4, stride=2, pad=1, initialW=w)
            self.c3_0 = L.Convolution2D(512, 512, 3, stride=1, pad=1, initialW=w)
            self.l4 = L.Linear(4 * 4 * 512, 100, initialW=w)
            self.bn0_1 = L.BatchNormalization(128, use_gamma=False)
            self.bn1_0 = L.BatchNormalization(128, use_gamma=False)
            self.bn1_1 = L.BatchNormalization(256, use_gamma=False)
            self.bn2_0 = L.BatchNormalization(256, use_gamma=False)
            self.bn2_1 = L.BatchNormalization(512, use_gamma=False)
            self.bn3_0 = L.BatchNormalization(512, use_gamma=False)

    def __call__(self, x):
        h = F.leaky_relu(self.c0_0(x))
        h = F.leaky_relu(self.bn0_1(self.c0_1(h)))
        h = F.leaky_relu(self.bn1_0(self.c1_0(h)))
        h = F.leaky_relu(self.bn1_1(self.c1_1(h)))
        h = F.leaky_relu(self.bn2_0(self.c2_0(h)))
        h = F.leaky_relu(self.bn2_1(self.c2_1(h)))
        h = F.leaky_relu(self.bn3_0(self.c3_0(h)))
        h = self.l4(h)
        return h


class EncoderGenerator(chainer.Chain):

    def __init__(self, wscale=0.02):
        super(EncoderGenerator, self).__init__()
        self.n_hidden = 100

        with self.init_scope():
            # Encoder
            w = chainer.initializers.Normal(wscale)
            self.c0_0 = L.Convolution2D(3, 64, 3, stride=2, pad=1, initialW=w)
            self.c0_1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, initialW=w)
            self.c1_0 = L.Convolution2D(128, 128, 3, stride=1, pad=1, initialW=w)
            self.c1_1 = L.Convolution2D(128, 256, 4, stride=2, pad=1, initialW=w)
            self.c2_0 = L.Convolution2D(256, 256, 3, stride=1, pad=1, initialW=w)
            self.c2_1 = L.Convolution2D(256, 512, 4, stride=2, pad=1, initialW=w)
            self.c3_0 = L.Convolution2D(512, 512, 3, stride=1, pad=1, initialW=w)
            self.l4 = L.Linear(4 * 4 * 512, 100, initialW=w)
            self.bn0_1 = L.BatchNormalization(128, use_gamma=False)
            self.bn1_0 = L.BatchNormalization(128, use_gamma=False)
            self.bn1_1 = L.BatchNormalization(256, use_gamma=False)
            self.bn2_0 = L.BatchNormalization(256, use_gamma=False)
            self.bn2_1 = L.BatchNormalization(512, use_gamma=False)
            self.bn3_0 = L.BatchNormalization(512, use_gamma=False)
            # Generator
            self.l0 = L.Linear(100, 4 * 4 * 512, initialW=w)
            self.dc1 = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, initialW=w)
            self.dc2 = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, initialW=w)
            self.dc3 = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, initialW=w)
            self.dc4 = L.Deconvolution2D(64, 3, 4, stride=2, pad=1, initialW=w)
            self.bn0 = L.BatchNormalization(4 * 4 * 512)
            self.bn1 = L.BatchNormalization(256)
            self.bn2 = L.BatchNormalization(128)
            self.bn3 = L.BatchNormalization(64)

    def make_hidden(self, batchsize):
        return numpy.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1))\
            .astype(numpy.float32)

    def __call__(self, x):
        # Encoder
        h = F.leaky_relu(self.c0_0(x))
        h = F.leaky_relu(self.bn0_1(self.c0_1(h)))
        h = F.leaky_relu(self.bn1_0(self.c1_0(h)))
        h = F.leaky_relu(self.bn1_1(self.c1_1(h)))
        h = F.leaky_relu(self.bn2_0(self.c2_0(h)))
        h = F.leaky_relu(self.bn2_1(self.c2_1(h)))
        h = F.leaky_relu(self.bn3_0(self.c3_0(h)))
        h = self.l4(h)
        # Generator
        h = F.reshape(F.leaky_relu(self.bn0(self.l0(h))), (len(h), 512, 4, 4))
        h = F.leaky_relu(self.bn1(self.dc1(h)))
        h = F.leaky_relu(self.bn2(self.dc2(h)))
        h = F.leaky_relu(self.bn3(self.dc3(h)))
        x = F.sigmoid(self.dc4(h))
        return x
