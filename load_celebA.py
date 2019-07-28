# ---author : xiaoyang---
# coding : utf-8

import pickle
import os
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import re
import argparse

ap=argparse.ArgumentParser()
ap.add_argument('--image',required=True,
                help='Path of images')
ap.add_argument('--bbox',required=True,
                help='Path of bounding boxes')
ap.add_argument('--attr',required=True,
                help='Path of attribute labels')
args=vars(ap.parse_args())

def get_crop_img(bbox_path,img_path,attr_path): #crop the images from the bounding box in them
    with open(bbox_path,'r') as bbox_file:
        with open(attr_path,'r') as attr_file:
            img_list = os.listdir(img_path)
            img_list.sort()
            num_of_img = len(img_list)
            bbox_list = bbox_file.readlines()
            attr_list = attr_file.readlines()
            X_smile = []
            X_gender = []
            X_glasses = []
            for i in range(num_of_img):
                img_name = os.path.join(img_path,img_list[i])
                img = cv2.imread(img_name)
                face_bbox = bbox_list[i+2]
                x = face_bbox.split()[1]
                y = face_bbox.split()[2]
                width = face_bbox.split()[3]
                height = face_bbox.split()[4]
                img = img[y:y+height,x:x+width]
                if img.size ==0:
                    continue
                print(img.size)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img,(48,48))
                X_smile.append((img,attr_list[i+2].split()[31]))
                X_glasses.append((img,attr_list[i+2].split()[15]))
                X_gender.append((img,attr_list[i+2].split()[20]))
                print(img_list[i])
                print(attr_list[i+2].split()[31])
                print(attr_list[i+2].split()[15])
                print(attr_list[i+2].split()[20])
            for _ in range(10):
                np.random.shuffle(X_smile)
                np.random.shuffle(X_gender)
                np.random.shuffle(X_glasses)
            print('data size:%d'%len(X_smile))
            train_data_smile, test_data_smile = X_smile[:160000], X_smile[160000:]
            train_data_gender, test_data_gender = X_gender[:160000], X_gender[160000:]
            train_data_glasses, test_data_glasses = X_glasses[:160000],X_glasses[160000:]
            np.save('./data/' + 'train_smile.npy', train_data_smile)
            np.save('./data/' + 'data_smile.npy', X_smile)
            np.save('./data/' + 'test_smile.npy', test_data_smile)
            np.save('./data/' + 'train_glasses.npy', train_data_glasses)
            np.save('./data/' + 'data_glasses.npy', X_glasses)
            np.save('./data/' + 'test_glasses.npy', test_data_glasses)
            np.save('./data/' + 'train_gender.npy', train_data_gender)
            np.save('./data/' + 'data_gender.npy', X_gender)
            np.save('./data/' + 'test_gender.npy', test_data_gender)

if __name__ == '__main__':
    get_crop_img(args['bbox'],args['image'],args['attr'])



