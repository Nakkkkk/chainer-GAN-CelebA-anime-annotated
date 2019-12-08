#!/usr/bin/python
# coding: UTF-8

# author Nakkkkk(https://github.com/Nakkkkk)

import sys, os
import cv2
import numpy as np
from natsort import natsorted

####################################
# Chainerデータセット用のパスリストを作成
####################################
def make_daraset_list(path):

    def find_all_files(directory):
        for root, dirs, files in os.walk(directory):
            yield root
            for file in files:
                yield os.path.join(root, file)

    file_name_list = []
    for file in find_all_files(path):
        file_name_list.append(file)

    img_path = []
    for file_name in file_name_list:
        if ".png" in file_name:
            img_path.append(file_name)
        elif ".jpg" in file_name:
            img_path.append(file_name)

    return natsorted(img_path)


###############################################
# Chainerデータセット用のパスリストを作成&annotation
###############################################
def make_daraset_list_with_annotation(path):

    def find_all_files(directory):
        for root, dirs, files in os.walk(directory):
            yield root
            for file in files:
                yield os.path.join(root, file)

    def find_all_files_label(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                yield root.split(directory+"/")[-1]

    file_name_list = []
    for file in find_all_files(path):
        file_name_list.append(file)

    file_label_list = []
    for label in find_all_files_label(path):
        file_label_list.append(label)

    img_path = []
    img_label = []

    print(" ")
    print("### file_name ###")
    for f in file_name_list[:3]:
        print(f)
    print(" ")
    print("### file_label ###")
    for f in file_label_list[:3]:
        print(f)

    for file_name, file_label in zip(file_name_list, file_label_list):
        if ".png" in file_name:
            img_path.append(file_name)
            img_label.append(file_label)
        elif ".jpg" in file_name:
            img_path.append(file_name)
            img_label.append(file_label)

    return img_path, img_label


##############################################
# Chainerデータセットannotation用のパスリストを作成
##############################################
def make_annotation_list(path):

    def find_all_files(directory):
        for root, dirs, files in os.walk(directory):
            yield root
            for file in files:
                yield os.path.join(root, file)

    file_name_list = []
    for file in find_all_files(path):
        file_name_list.append(file)

    img_path = []
    for file_name in file_name_list:
        if ".txt" in file_name:
            img_path.append(file_name)

    img_path = natsorted(img_path)
    for path in img_path:
        print(path)

    annotation = []
    annotation_type = ""
    annotation_type_list = []
    for path in img_path:
        with open(path) as f:
            l = f.readlines()
            annotation += l[2:]
            annotation_type = l[1]
    annotation_type_list = annotation_type.split()

    for i in range(len(annotation)):
        annotation[i] = annotation[i].split("\r\n")[0]

    annotation_list = []
    for i in range(len(annotation)):
        annotation_split = annotation[i].split()
        tmp_ann = [annotation_split[0]]
        for j in range(1, len(annotation_split)):
            # anime_annotationのラベルをCelabA基準にする。1は該当、-1は非該当、0は不明。
            if "dataset_celebA_label_anime" in path:
                if annotation_split[j] == "-1":
                    annotation_split[j] = "0"
                elif annotation_split[j] == "0":
                    annotation_split[j] = "-1"
            tmp_ann.append(int(annotation_split[j]))

        if len(annotation_split) > 2:
            annotation_list.append(tmp_ann)

    return annotation_list, annotation_type_list


##################################################################
# Chainerデータセットのannotationのtypeを、複数のデータセット間で統一する。
##################################################################
def organize_annotation_type(*args):

    print("num of file = " + str(len(args)))
    print(" ")
    print("### anime before unify ###")
    for a in args[0][0][:3]:
        print(a)
    print(" ")
    print("### human before unify ###")
    for a in args[1][0][:3]:
        print(a)

    # すべてのannotationに共通するstandard_annotation_typeを求める
    standard_annotation_type = args[0][1]
    for i in range(1, len(args)):
        tmp_standard_annotation_type = []
        for j in range(len(args[i][1])):
            for k in range(len(standard_annotation_type)):
                if standard_annotation_type[k] == args[i][1][j]:
                    tmp_standard_annotation_type.append(args[i][1][j])
        standard_annotation_type = tmp_standard_annotation_type
    print(" ")
    print("### standard_annotation_type ###")
    print(standard_annotation_type)

    # 統一後に、それぞれのannotationのtypeの順番となる配列を作成
    each_annotation_type_orders = []
    for i in range(len(args)):
        each_annotation_type_orders.append([])
        for j in range(len(standard_annotation_type)):
            each_annotation_type_orders[-1].append(-1)

    # それぞれのannotationのtypeとstandard_annotation_typeを比較して順番を並び替える
    for i in range(len(args)):
        for j in range(len(standard_annotation_type)):
            for k in range(len(args[i][1])):
                if standard_annotation_type[j] == args[i][1][k]:
                    each_annotation_type_orders[i][j] = k
    print(" ")
    print("### each_annotation_type_orders ###")
    print(each_annotation_type_orders)

    # annotationのtypeを並べ替える
    dataset_unified = []
    for i in range(len(args)):
        dataset_unified.append([])
        for j in range(len(args[i][0])):
            dataset_unified[i].append([args[i][0][j][0]])
            for k in range(len(each_annotation_type_orders[i])):
                dataset_unified[i][j].append(args[i][0][j][each_annotation_type_orders[i][k] + 1])

    print(" ")
    print("### anime after unify ###")
    for d in dataset_unified[0][:3]:
        print(d)
    for d in dataset_unified[0][-3:]:
        print(d)
    print(" ")
    print("### human after unify ###")
    for d in dataset_unified[1][:3]:
        print(d)
    for d in dataset_unified[1][-3:]:
        print(d)

    return dataset_unified, standard_annotation_type


##############################################
# Chainerデータセットをannotation
##############################################
def annotate_dataset(datas, annotations):

    print("")
    print("### before annotated ###")
    print("len datas = " + str(len(datas)))
    print("len annotations = " + str(len(annotations)))
    for i in range(len(annotations[:3])):
        print(annotations[i][0])
        print(datas[i])
    for i in range(len(annotations[-3:])):
        print(annotations[-i-1][0])
        print(datas[-i-1])

    annotated_dataset = []
    if len(datas) == len(annotations):
        if annotations[0][0] in datas[0] and annotations[-1][0] in datas[-1]:
            print("annotation and datas may be one to one !!")
            print(" ")
            print("len(datas) == len(annotations) (human)")
            for data, annotation in zip(datas, annotations):
                annotated_dataset.append((data, annotation[1:]))
        else:
            print("annotation and datas is not one to one...")

    elif len(datas) > len(annotations):
        print(" ")
        print("len(datas) > len(annotations) (anime)")

        for data in datas:
            for annotation in annotations:
                if annotation[0] in data:
                    annotated_dataset.append((data, annotation[1:]))

    print(" ")
    print("### after annotated ###")
    for i in range(len(annotated_dataset[:3])):
        print(annotated_dataset[:3][i])
    for i in range(len(annotated_dataset[-3:])):
        print(annotated_dataset[-3:][i])

    return annotated_dataset


if __name__ == '__main__':

    # アニメ顔データセットの取得
    anime_img_path = "./../../anime_face/animeface-character-dataset-resize160x160/thumb"
    anime_img_files = make_daraset_list(anime_img_path)
    # 人間顔データセットの取得
    human_img_path = "./../../human_face/img_align_celeba"
    human_img_files = make_daraset_list(human_img_path)
    # アニメ顔データセットのラベルの取得
    anime_ann_path = "./../../anime_face/dataset_celebA_label_anime"
    anime_ann_files, anime_ann_type = make_annotation_list(anime_ann_path)
    # 人間顔データセットのラベルの取得
    human_ann_path = "./../../anime_face/dataset_celebA_label_human"
    human_ann_files, human_ann_type = make_annotation_list(human_ann_path)


    dataset_unified, standard_annotation_type = organize_annotation_type([anime_ann_files, anime_ann_type], [human_ann_files, human_ann_type])

    anime_img_files_annotated = annotate_dataset(anime_img_files, dataset_unified[0])
    human_img_files_annotated = annotate_dataset(human_img_files, dataset_unified[1])



