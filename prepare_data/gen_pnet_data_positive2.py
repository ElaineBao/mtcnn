import numpy as np
import cv2
import os
import math
import numpy.random as npr
from utils import IoU
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


DEBUG = False
def get_offsets(pts, im_center):
    offsets = []
    for idx, pt in enumerate(pts):
        if idx % 2 == 0: # x axis
            offset_x = (pt - im_center[0]) / im_center[0]
            offsets.append(offset_x)
        else: # y axis
            offset_y = (pt - im_center[1]) / im_center[1]
            offsets.append(offset_y)

    return offsets

def get_box_coords(offsets, im_center):
    box_coords = []
    for idx, offset in enumerate(offsets):
        if idx % 2 == 0:
            box_coord_x = offset * im_center[0] + im_center[0]
            box_coords.append(box_coord_x)
        else:
            box_coord_y = offset * im_center[1] + im_center[1]
            box_coords.append(box_coord_y)

    return box_coords


def generate_pnet_data_positive(anno_file, background_file, anno_img_dir, img_save_dir, list_save_dir):
    if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)
    f1 = open(os.path.join(list_save_dir, 'train2.txt'), 'w')

    with open(anno_file, 'r') as f:
        annotations = f.readlines()
    with open(background_file, 'r') as f:
        background_annotations = f.readlines()

    num_images = len(annotations)
    num_backgrounds = len(background_annotations)
    print "%d pics in total" % num_images
    positive_idx = 0 # positive
    img_idx = 0
    for annotation in annotations:
        img_idx += 1
        annotation = annotation.strip().split(' ')
        im_path = annotation[0]
        #im_path = os.path.join(anno_img_dir, im_path)
        offsets = map(float, annotation[2:])
        offsets = np.array(offsets, dtype=np.float32).reshape(-1, 8)
        img = cv2.imread(im_path)
        if img == None:
            print "cannot find image or img is broken: {}".format(im_path)
            continue

        im_height, im_width, im_channel = img.shape
        im_center = (im_width/2, im_height/2)

        for box_offsets in offsets:
            box = get_box_coords(box_offsets, im_center)
            if DEBUG:
                img_debug = np.zeros(img.shape, np.uint8)
                img_debug = img.copy()
                cv2.line(img_debug, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 255, 5)
                cv2.line(img_debug, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), 255, 5)
                cv2.imwrite("debug/%s_origin.jpg" % positive_idx, img_debug)

            # mask with write background
            fg_mask = np.full((im_height, im_width), 0, dtype=np.uint8)
            roi_corners = np.array([[(box[0], box[1]), (box[2], box[3]), (box[4], box[5]), (box[6], box[7])]], dtype=np.int32)
            # fill the ROI so it doesn't get wiped out when the mask is applied
            content_color = (255,) * im_channel
            cv2.fillConvexPoly(fg_mask, roi_corners, content_color)
            # get first masked value (foreground)
            fg = cv2.bitwise_or(img, img, mask=fg_mask)

            # get second masked value (background) mask must be inverted
            bg_mask = cv2.bitwise_not(fg_mask)
            background = np.full(img.shape, 255, dtype=np.uint8)
            bk = cv2.bitwise_or(background, background, mask=bg_mask)

            # combine foreground+background
            masked_img = cv2.bitwise_or(fg, bk)
            save_file = os.path.join(img_save_dir, "%s.jpg" % positive_idx)
            cv2.imwrite(save_file, masked_img)
            f1.write("%s" % os.path.join(img_save_dir,
                                     str(positive_idx) + '.jpg') + ' 1 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n' % (
                                    box_offsets[0], box_offsets[1], box_offsets[2], box_offsets[3],
                                    box_offsets[4], box_offsets[5], box_offsets[6], box_offsets[7]))

            positive_idx += 1


            # mask with arbitrary background
            fg_mask = np.full((im_height, im_width), 0, dtype=np.uint8)
            roi_corners = np.array([[(box[0], box[1]), (box[2], box[3]), (box[4], box[5]), (box[6], box[7])]],
                                   dtype=np.int32)
            # fill the ROI so it doesn't get wiped out when the mask is applied
            content_color = (255,) * im_channel
            cv2.fillConvexPoly(fg_mask, roi_corners, content_color)
            # get first masked value (foreground)
            fg = cv2.bitwise_or(img, img, mask=fg_mask)

            bg_num = npr.randint(0, num_backgrounds, 3)
            bg_mask = cv2.bitwise_not(fg_mask)
            for bg_idx in bg_num:
                bg_file = background_annotations[bg_idx].strip().split(' ')[0]
                background = cv2.imread(bg_file)
                background = cv2.resize(background, (im_width, im_height))
                bk = cv2.bitwise_or(background, background, mask=bg_mask)

                # combine foreground+background
                masked_img = cv2.bitwise_or(fg, bk)
                save_file = os.path.join(img_save_dir, "%s.jpg" % positive_idx)
                cv2.imwrite(save_file, masked_img)
                f1.write("%s" % os.path.join(img_save_dir,
                                             str(positive_idx) + '.jpg') + ' 1 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n' % (
                             box_offsets[0], box_offsets[1], box_offsets[2], box_offsets[3],
                             box_offsets[4], box_offsets[5], box_offsets[6], box_offsets[7]))

                positive_idx += 1


        print "%s/%s images done, positive samples: %s" % (img_idx, num_images, positive_idx)

    f1.close()

if __name__ == '__main__':
    anno_file = "data/IDCard/imglists/train.txt"
    background_file = "data/imagenet/imglists/train.txt"
    anno_img_dir = "data/IDCard/images"
    img_save_dir = "data/IDCard/generate_positive2"
    list_save_dir = "data/IDCard/imglists/"

    generate_pnet_data_positive(anno_file, background_file, anno_img_dir, img_save_dir, list_save_dir)
