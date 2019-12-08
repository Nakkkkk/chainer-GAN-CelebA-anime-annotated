# coding: UTF-8

# original code URL https://github.com/xkumiyu/chainer-GAN-CelebA
# revised by Nakkkkk(https://github.com/Nakkkkk)

import os

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable


def out_generated_image(gen, dis, rows, cols, seed, dst, img_paths, img_size):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        xp = gen.xp
        # 入力画像の整形
        img_animes = []
        for img_path in img_paths:
            img_anime = Image.open(img_path)
            img_anime = img_anime.resize(img_size)
            img_anime = np.asarray(img_anime, dtype=np.float32)
            img_anime = img_anime.transpose(2, 0, 1) # PILのImageは(height, width, channel)なのでChainerの形式に変換
            img_animes.append(img_anime)
        z = Variable(xp.asarray(img_animes)) / 255.

        with chainer.using_config('train', False):
            x = gen(z)
        x = chainer.cuda.to_cpu(x.data)
        np.random.seed()

        x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
        _, _, H, W = x.shape
        x = x.reshape((rows, cols, 3, H, W))
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape((rows * H, cols * W, 3))

        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir +\
            '/image{:0>8}.png'.format(trainer.updater.iteration)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        Image.fromarray(x).save(preview_path)

    return make_image
