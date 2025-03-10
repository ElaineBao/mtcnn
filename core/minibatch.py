import cv2
from tools import image_processing
from config import config
import numpy as np

def get_minibatch(imdb, num_classes, im_size):
    # im_size: 12, 24 or 48
    num_images = len(imdb)
    processed_ims = list()
    cls_label = list()
    bbox_reg_target = list()
    for i in range(num_images):
        im = cv2.imread(imdb[i]['image'])
        im = cv2.resize(im,(im_size, im_size))
        h, w, c = im.shape
        cls = imdb[i]['label']
        bbox_target = imdb[i]['bbox_target']

        assert h == w == im_size, "image size wrong"
        if imdb[i]['flipped']:
            im = im[:, ::-1, :]

        im_tensor = image_processing.transform(im)
        processed_ims.append(im_tensor)
        cls_label.append(cls)
        bbox_reg_target.append(bbox_target)

    im_array = np.vstack(processed_ims)
    label_array = np.array(cls_label)
    bbox_target_array = np.vstack(bbox_reg_target)
    '''
    bbox_reg_weight = np.ones(label_array.shape)
    invalid = np.where(label_array == 0)[0]
    bbox_reg_weight[invalid] = 0
    bbox_reg_weight = np.repeat(bbox_reg_weight, 4, axis=1)
    '''
    if im_size == 12:
        label_array = label_array.reshape(-1, 1)

    data = {'data': im_array}
    label = {'label': label_array,
             'bbox_target': bbox_target_array}

    return data, label

def get_testbatch(imdb):
    assert len(imdb) == 1, "Single batch only"
    im = cv2.imread(imdb[0]['image'])
    im_array = im
    data = {'data': im_array}
    label = {}
    return data, label
