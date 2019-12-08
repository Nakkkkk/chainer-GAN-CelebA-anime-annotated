#!/usr/bin/python
# coding: UTF-8

# original code URL https://github.com/xkumiyu/chainer-GAN-CelebA
# revised by Nakkkkk(https://github.com/Nakkkkk)

import chainer
import chainer.functions as F
from chainer import Variable

import time
import cv2
import matplotlib.pyplot as plt

class DCGANUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self.k = 0
        super(DCGANUpdater, self).__init__(*args, **kwargs)

    def loss_dis(self, dis, y_fake, y_real, y_fake_annotation, y_real_annotation):
        batchsize = len(y_fake)
        L1 = F.sum(F.softplus(-y_real[:,0])) / batchsize
        L2 = F.sum(F.softplus(y_fake[:,0])) / batchsize
        anno_loss = F.mean_squared_error(y_fake[:,1:], y_fake_annotation) + F.mean_squared_error(y_real[:,1:], y_real_annotation)
        loss = L1 + L2 + anno_loss
        chainer.report({'loss': loss}, dis)
        return loss

    def loss_gen(self, gen, y_fake, y_fake_annotation):
        batchsize = len(y_fake)
        L1 = F.sum(F.softplus(-y_fake[:,0])) / batchsize
        anno_loss = F.mean_squared_error(y_fake[:,1:], y_fake_annotation)
        loss = L1 + anno_loss
        chainer.report({'loss': loss}, gen)
        return loss

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        # iteratorを画像データとアノテーションデータに分割
        batch = self.get_iterator('main').next()
        batch_anime = []
        batch_human = []
        batch_anime_annotation = []
        batch_human_annotation = []
        for batch_idx in range(len(batch)):
            for batch_idx_2 in range(len(batch[batch_idx])):
                if batch_idx_2 == 0:
                    batch_anime.append(batch[batch_idx][batch_idx_2])
                elif batch_idx_2 == 1:
                    batch_human.append(batch[batch_idx][batch_idx_2])
                elif batch_idx_2 == 2:
                    batch_anime_annotation.append(batch[batch_idx][batch_idx_2])
                elif batch_idx_2 == 3:
                    batch_human_annotation.append(batch[batch_idx][batch_idx_2])

        # 人間顔データの整形
        x_real = Variable(self.converter(batch_human, self.device)) / 255.
        x_real = F.resize_images(x_real, (64, 64))
        xp = chainer.cuda.get_array_module(x_real.data)
        # GenとDisのセットアップ
        gen, dis = self.gen, self.dis
        batchsize = len(batch_human)
        # 人間顔アノテーションデータの整形
        y_real = dis(x_real)
        y_real_annotation = Variable(self.converter(batch_human_annotation, self.device))
        # フェイク画像（アニメ→人間）の生成
        x = Variable(self.converter(batch_anime, self.device)) / 255.
        x_fake = gen(x)
        y_fake = dis(x_fake)
        y_fake_annotation = Variable(self.converter(batch_anime_annotation, self.device))
        # dis_optimizerの更新
        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real, y_fake_annotation, y_real_annotation)
 
        # モード崩壊を回避する学習ルーチン（https://qiita.com/xkumiyu/items/1cc0223486c560062e00）
        if self.k == 0:
            # unrolling_stepsまではDisの重みを保存
            dis.cache_discriminator_weights()
        if self.k == dis.unrolling_steps:
            # gen_optimizerの更新
            gen_optimizer.update(self.loss_gen, gen, y_fake, y_fake_annotation)
            # unrolling_stepsからは保存したDisの重みを上書き
            dis.restore_discriminator_weights()
            self.k = -1
        self.k += 1


class EncUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.enc = kwargs.pop('models')
        super(EncUpdater, self).__init__(*args, **kwargs)

    def loss_enc(self, enc, x_real, x_fake):
        loss = F.mean_squared_error(x_real, x_fake)
        chainer.report({'loss': loss}, enc)
        return loss

    def update_core(self):
        enc_optimizer = self.get_optimizer('enc')

        batch = self.get_iterator('main').next()
        x_real = Variable(self.converter(batch, self.device)) / 255.
        x_real = F.resize_images(x_real, (64, 64))

        gen, enc = self.gen, self.enc
        z = enc(x_real)
        x_fake = gen(z)

        enc_optimizer.update(self.loss_enc, enc, x_real, x_fake)
