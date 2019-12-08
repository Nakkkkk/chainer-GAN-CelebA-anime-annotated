# coding: UTF-8

# original code URL https://github.com/xkumiyu/chainer-GAN-CelebA
# revised by Nakkkkk(https://github.com/Nakkkkk)

import chainer
import numpy
from PIL import Image

import random

class CelebADataset(chainer.dataset.DatasetMixin):
    def __init__(self, paths, size):
        self._paths = paths
        self._size = size
        self._dtype = numpy.float32
        print("len self._paths = " + str(len(self._paths)))
        print("self._size = " + str(self._size))
        print("self._dtype = " + str(self._dtype))

    def __len__(self):
        # データセットの数を返します
        return len(self._paths)

    def get_example(self, i):
        # データセットのインデックスを受け取って、データを返します
        img= Image.open(self._paths[i])
        img = img.resize(self._size) # PILをつかってリサイズ
        img = numpy.asarray(img, dtype=self._dtype) # float32型のnumpy arrayに変換
        img = img.transpose(2, 0, 1) # PILのImageは(height, width, channel)なのでChainerの形式に変換
        #print(img.shape)
        return img

class AnimeDataset(chainer.dataset.DatasetMixin):
    def __init__(self, paths, size):
        self._paths = paths
        self._size = size
        self._dtype = numpy.float32
        print("len self._paths = " + str(len(self._paths)))
        print("self._size = " + str(self._size))
        print("self._dtype = " + str(self._dtype))

    def __len__(self):
        # データセットの数を返します
        return len(self._paths)

    def get_example(self, i):
        # データセットのインデックスを受け取って、データを返します
        # anime dataset
        img_anime = Image.open(self._paths[i][0])
        img_anime = img_anime.resize(self._size) # PILをつかってリサイズ
        img_anime = numpy.asarray(img_anime, dtype=self._dtype) # float32型のnumpy arrayに変換
        img_anime = img_anime.transpose(2, 0, 1) # PILのImageは(height, width, channel)なのでChainerの形式に変換
        # human dataset
        img_human = Image.open(self._paths[i][1])
        img_human = img_human.resize(self._size) # PILをつかってリサイズ
        img_human = numpy.asarray(img_human, dtype=self._dtype) # float32型のnumpy arrayに変換
        img_human = img_human.transpose(2, 0, 1) # PILのImageは(height, width, channel)なのでChainerの形式に変換

        return img_anime, img_human


class AnimeDataset_annotated(chainer.dataset.DatasetMixin):
    def __init__(self, anime_img_files_annotated, human_img_files_annotated, size):
        self._anime_img_files_annotated = anime_img_files_annotated
        self._human_img_files_annotated = human_img_files_annotated
        self._size = size
        self._dtype = numpy.float32
        print("len self._anime_img_files_annotated = " + str(len(self._anime_img_files_annotated)))
        print("len self._human_img_files_annotated = " + str(len(self._human_img_files_annotated)))
        print("self._size = " + str(self._size))
        print("self._dtype = " + str(self._dtype))
        self._human_img_files_annotated = random.sample(self._human_img_files_annotated, len(self._anime_img_files_annotated))
        print("len self._human_img_files_annotated = " + str(len(self._human_img_files_annotated)))

    def __len__(self):
        # データセットの数を返します
        # _anime_img_files_annotatedの長さを基準にする。
        return len(self._anime_img_files_annotated)

    def get_example(self, i):
        # データセットのインデックスを受け取って、データを返します
        # anime dataset
        img_anime = Image.open(self._anime_img_files_annotated[i][0])
        img_anime = img_anime.resize(self._size) # PILをつかってリサイズ
        img_anime = numpy.asarray(img_anime, dtype=self._dtype) # float32型のnumpy arrayに変換
        img_anime = img_anime.transpose(2, 0, 1) # PILのImageは(height, width, channel)なのでChainerの形式に変換
        # human dataset
        img_human = Image.open(self._human_img_files_annotated[i][0])
        img_human = img_human.resize(self._size) # PILをつかってリサイズ
        img_human = numpy.asarray(img_human, dtype=self._dtype) # float32型のnumpy arrayに変換
        img_human = img_human.transpose(2, 0, 1) # PILのImageは(height, width, channel)なのでChainerの形式に変換

        img_anime_annotation = numpy.asarray(self._anime_img_files_annotated[i][1], dtype=self._dtype)
        img_human_annotation = numpy.asarray(self._human_img_files_annotated[i][1], dtype=self._dtype)


        return img_anime, img_human, img_anime_annotation, img_human_annotation


