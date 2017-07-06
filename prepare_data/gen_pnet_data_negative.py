import numpy as np
import cv2
import os
import math
import numpy.random as npr
from utils import IoU
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def generate_pnet_data_negative(anno_file, anno_img_dir, list_save_dir):
    if not os.path.exists(list_save_dir):
        os.mkdir(list_save_dir)
    f1 = open(os.path.join(list_save_dir, 'train.txt'), 'w')

    with open(anno_file, 'r') as f:
        annotations = f.readlines()

    num_images = len(annotations)
    print "%d pics in total" % num_images
    negative_idx = 0 # positive
    img_idx = 0
    for annotation in annotations:
        img_idx += 1
        im_path = os.path.join(anno_img_dir, annotation.strip() + '.JPEG')
        img = cv2.imread(im_path)
        if img is None:
            print "cannot find image or img is broken: {}".format(im_path)
            continue

        f1.write("%s 0\n" % im_path)


        print "%s/%s images done" % (img_idx, num_images)

    f1.close()

if __name__ == '__main__':
    anno_file = "data/imagenet/ImageSets/DET/train.txt"
    anno_img_dir = "data/imagenet/Data/DET/train"
    list_save_dir = "data/imagenet/imglists/"

    generate_pnet_data_negative(anno_file, anno_img_dir,list_save_dir)
