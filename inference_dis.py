# coding: UTF-8

import chainer
import chainer.links as L

from dataset import CelebADataset, AnimeDataset
from net import Discriminator
from chainer import Variable

import numpy as np
from PIL import Image
import my_lib
from net import EncoderGenerator
import random

test_model_num = 0

# EncoderGenerator
if test_model_num == 0:


    infer_net = Discriminator()
    chainer.serializers.load_npz(
        '/home/dl-box/teamviewer/nk/chainer-GAN-CelebA-anime-annotated/result/dis_iter_65000.npz',
        infer_net, strict=False)

    gpu_id = 0
    if gpu_id >= 0:
        infer_net.to_gpu(gpu_id)

    anime_img_files_rand_N = ["/home/dl-box/teamviewer/nk/chainer-GAN-CelebA-anime-annotatedv2/gen40000_imgs/bad05_d/out_image.png"]

    rows = 1
    cols = 1
    seed = 0
    img_size = (64,64)
    xp = infer_net.xp

    img_animes = []
    for img_path in anime_img_files_rand_N:
        img_anime = Image.open(img_path)
        img_anime = img_anime.resize(img_size) # PILをつかってリサイズ
        img_anime = np.asarray(img_anime, dtype=np.float32) # float32型のnumpy arrayに変換
        img_anime = img_anime.transpose(2, 0, 1) # PILのImageは(height, width, channel)なのでChainerの形式に変換
        img_animes.append(img_anime)
    z = Variable(xp.asarray(img_animes)) / 255.
    #print("z shape = " + str(z.shape))



with chainer.using_config('train', False):
    x = infer_net(z)
x = chainer.cuda.to_cpu(x.data)
np.random.seed()
print("path = " + str(anime_img_files_rand_N))
print("x = " + str(x))

