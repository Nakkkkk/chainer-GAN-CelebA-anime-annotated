#!/usr/bin/python
# coding: UTF-8

# original code URL https://github.com/xkumiyu/chainer-GAN-CelebA
# revised by Nakkkkk(https://github.com/Nakkkkk)

import argparse
import os


import matplotlib
try:
    matplotlib.use('Agg')
except Exception:
    raise

import chainer
from chainer import training
from chainer.training import extensions

from dataset import CelebADataset, AnimeDataset, AnimeDataset_annotated
from net import Discriminator
from updater import DCGANUpdater
from visualize import out_generated_image

import my_lib
from net import EncoderGenerator
import random

def main():
    parser = argparse.ArgumentParser(description='Train GAN')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=500,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default='.',
                        help='Directory of image files.')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
    parser.add_argument('--snapshot_interval', type=int, default=5000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=1000,
                        help='Interval of displaying log to console')
    parser.add_argument('--unrolling_steps', type=int, default=0)
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# batchsize: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # NNのセットアップ
    gen = EncoderGenerator()
    dis = Discriminator(unrolling_steps=args.unrolling_steps)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()

    # optimizerのセットアップ
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        return optimizer
    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)

    # データセットのセットアップ
    # アニメ顔データセットの取得
    anime_img_path = "./../../anime_face/animeface-character-dataset-resize160x160/thumb"
    anime_img_files = my_lib.make_daraset_list(anime_img_path)
    # 人間顔データセットの取得
    human_img_path = "./../../human_face/img_align_celeba"
    human_img_files = my_lib.make_daraset_list(human_img_path)
    # アニメ顔データセットのラベルの取得
    anime_ann_path = "./../../anime_face/dataset_celebA_label_anime"
    anime_ann_files, anime_ann_type = my_lib.make_annotation_list(anime_ann_path)
    # 人間顔データセットのラベルの取得
    human_ann_path = "./../../anime_face/dataset_celebA_label_human"
    human_ann_files, human_ann_type = my_lib.make_annotation_list(human_ann_path)
    # 顔データにアノテーションを付与
    dataset_unified, standard_annotation_type = my_lib.organize_annotation_type([anime_ann_files, anime_ann_type], [human_ann_files, human_ann_type])
    anime_img_files_annotated = my_lib.annotate_dataset(anime_img_files, dataset_unified[0])
    human_img_files_annotated = my_lib.annotate_dataset(human_img_files, dataset_unified[1])
    # アノテーション付きデータセットを生成
    train_anime = AnimeDataset_annotated(anime_img_files_annotated, human_img_files_annotated, size=(64, 64))
    train_iter_anime = chainer.iterators.SerialIterator(train_anime, args.batchsize)

    # Trainerのセットアップ
    updater = DCGANUpdater(
        models=(gen, dis),
        iterator=train_iter_anime,
        optimizer={
            'gen': opt_gen, 'dis': opt_dis},
        device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    trainer.extend(
        extensions.snapshot(filename='snapshot_gan_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval, log_name='train_gan.log'))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'gen/loss', 'dis/loss',
    ]), trigger=display_interval)
    trainer.extend(extensions.PlotReport(
        ['gen/loss', 'dis/loss'], trigger=display_interval, file_name='gan-loss.png'))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    anime_img_files_rand_N = random.sample(anime_img_files, 100)
    trainer.extend(
        out_generated_image(
            gen, dis,
            10, 10, args.seed, args.out, anime_img_files_rand_N, (64,64)),
        trigger=snapshot_interval)

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
