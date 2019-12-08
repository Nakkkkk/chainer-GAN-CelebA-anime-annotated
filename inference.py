# coding: UTF-8

# author Nakkkkk(https://github.com/Nakkkkk)

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


# EncoderGeneratorのセットアップ
infer_net = EncoderGenerator()
chainer.serializers.load_npz(
    './result/gen_iter_40000.npz',
    infer_net, strict=False)

gpu_id = 0
if gpu_id >= 0:
    infer_net.to_gpu(gpu_id)

# 入力画像の選定（ランダムに１枚）
anime_img_path = "./../../anime_face/animeface-character-dataset-resize160x160/thumb"
anime_img_files = my_lib.make_daraset_list(anime_img_path)
anime_img_files_rand_N = random.sample(anime_img_files, 1)

# 各種パラメータ
rows = 1
cols = 1
seed = 0
img_size = (64,64)
xp = infer_net.xp

# 入力画像の整形
img_animes = []
for img_path in anime_img_files_rand_N:
    img_anime = Image.open(img_path)
    img_anime = img_anime.resize(img_size)
    img_anime = np.asarray(img_anime, dtype=np.float32)
    img_anime = img_anime.transpose(2, 0, 1) # PILのImageは(height, width, channel)なのでChainerの形式に変換
    img_animes.append(img_anime)
z = Variable(xp.asarray(img_animes)) / 255.
# 入力アニメ画像から人間画像を出力
with chainer.using_config('train', False):
    x = infer_net(z)
x = chainer.cuda.to_cpu(x.data)
z = chainer.cuda.to_cpu(z.data)
np.random.seed()

# 入力画像の整形
x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
_, _, H, W = x.shape
x = x.reshape((rows, cols, 3, H, W))
x = x.transpose(0, 3, 1, 4, 2)
x = x.reshape((rows * H, cols * W, 3))
# 入力画像の保存
preview_dir = './'
preview_path = preview_dir +\
    '/out_image.png'
Image.fromarray(x).save(preview_path)

# 生成画像の整形
x = np.asarray(np.clip(z * 255, 0.0, 255.0), dtype=np.uint8)
_, _, H, W = x.shape
x = x.reshape((rows, cols, 3, H, W))
x = x.transpose(0, 3, 1, 4, 2)
x = x.reshape((rows * H, cols * W, 3))
# 生成画像の保存
preview_dir = './'
preview_path = preview_dir +\
    '/in_image.png'
Image.fromarray(x).save(preview_path)

# Discriminatorによる生成画像の属性判定
infer_net = Discriminator()
chainer.serializers.load_npz(
    './result/dis_iter_40000.npz',
    infer_net, strict=False)
with chainer.using_config('train', False):
    x = infer_net(z)
x = chainer.cuda.to_cpu(x.data)
np.random.seed()
print("path = " + str(anime_img_files_rand_N))
print("x = " + str(x))

