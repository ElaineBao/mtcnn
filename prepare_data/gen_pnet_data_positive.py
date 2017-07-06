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


def get_rotated_coordinates(pts, degree, im_center, im_center_rotated):
    assert len(pts) == 8, "num of box coordinates is not 8"
    pts_rotated = []
    angle = degree / 180.0 * math.pi
    for idx in range(0, 8, 2):
        x_rotated = (pts[idx] - im_center[0]) * math.cos(angle) + (pts[idx+1] - im_center[1]) * math.sin(angle) + im_center_rotated[0]
        y_rotated = -(pts[idx] - im_center[0]) * math.sin(angle) + (pts[idx+1] - im_center[1]) * math.cos(angle) + im_center_rotated[1]
        pts_rotated.append(x_rotated)
        pts_rotated.append(y_rotated)

    return pts_rotated

def img_rotation(img, im_center, degree):
    im_width = im_center[0] * 2
    im_height = im_center[1] * 2
    matRotation = cv2.getRotationMatrix2D(im_center, degree, 1.0)
    angle = degree / 180.0 * math.pi
    im_width_rotated = int(abs(im_width * math.cos(angle)) + abs(im_height * math.sin(angle)))
    im_height_rotated = int(abs(im_height * math.cos(angle)) + abs(im_width * math.sin(angle)))
    matRotation[0, 2] += (im_width_rotated - im_width) / 2
    matRotation[1, 2] += (im_height_rotated - im_height) / 2
    img_rotated = cv2.warpAffine(img, matRotation, (im_width_rotated, im_height_rotated))
    im_center_rotated = (im_width_rotated / 2, im_height_rotated / 2)

    return img_rotated, im_center_rotated

def generate_pnet_data_positive(anno_file, anno_img_dir, img_save_dir, list_save_dir):
    if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)
    f1 = open(os.path.join(list_save_dir, 'train.txt'), 'w')

    with open(anno_file, 'r') as f:
        annotations = f.readlines()

    num_images = len(annotations)
    print "%d pics in total" % num_images
    positive_idx = 0 # positive
    img_idx = 0
    for annotation in annotations:
        img_idx += 1
        annotation = annotation.strip().split(' ')
        im_path = annotation[0]
        im_path = os.path.join(anno_img_dir, im_path)
        bbox = map(float, annotation[1:])
        boxes = np.array(bbox, dtype=np.float32).reshape(-1, 8)
        img = cv2.imread(im_path)
        if img == None:
            print "cannot find image or img is broken: {}".format(im_path)
            continue

        im_height, im_width, im_channel = img.shape
        im_center = (im_width/2, im_height/2)
        for box in boxes:
            offsets = get_offsets(box, im_center)
            f1.write("%s" % im_path + ' 1 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n' % (offsets[0], offsets[1], offsets[2], offsets[3], offsets[4], offsets[5], offsets[6], offsets[7]))
            if DEBUG:
                img_debug = np.zeros(img.shape, np.uint8)
                img_debug = img.copy()
                cv2.line(img_debug, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 255, 5)
                cv2.line(img_debug, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), 255, 5)
                cv2.imwrite("debug/%s_origin.jpg" % positive_idx, img_debug)

        #generate positive samples via rotation -  90,180,270 degree and 4 random degree
        degrees = [90,180,270]
        degrees.extend(npr.randint(0, 360, size=4))
        for degree in degrees:
            img_rotated, im_center_rotated = img_rotation(img, im_center, degree)
            save_file = os.path.join(img_save_dir, "%s.jpg" % positive_idx)
            cv2.imwrite(save_file, img_rotated)
            for box in boxes:
                box_rotated = get_rotated_coordinates(box, degree, im_center, im_center_rotated)
                if DEBUG:
                    img_debug = np.zeros(img_rotated.shape, np.uint8)
                    img_debug = img_rotated.copy()
                    img_debug = cv2.line(img_debug,(int(box_rotated[0]),int(box_rotated[1])),(int(box_rotated[2]),int(box_rotated[3])), 255, 5)
                    img_debug = cv2.line(img_debug, (int(box_rotated[4]),int(box_rotated[5])), (int(box_rotated[6]),int(box_rotated[7])), 255, 5)
                    cv2.imwrite("debug/%s.jpg"%positive_idx, img_debug)
                offsets = get_offsets(box_rotated, im_center_rotated)
                f1.write("%s" % os.path.join(img_save_dir, str(positive_idx) + '.jpg') + ' 1 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n' % (offsets[0], offsets[1], offsets[2], offsets[3], offsets[4], offsets[5], offsets[6], offsets[7]))

            positive_idx += 1

        print "%s/%s images done, positive samples: %s" % (img_idx, num_images, positive_idx)

    f1.close()

if __name__ == '__main__':
    anno_file = "data/IDCard/imglists/gt.txt"
    anno_img_dir = "data/IDCard/images"
    img_save_dir = "data/IDCard/generate_positive2"
    list_save_dir = "data/IDCard/imglists/"

    generate_pnet_data_positive(anno_file, anno_img_dir, img_save_dir, list_save_dir)
