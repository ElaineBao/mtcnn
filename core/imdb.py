import mxnet as mx
import os
import cPickle
import numpy as np
from config import config

class IMDB(object):
    def __init__(self, dataset, image_set, root_path, positive_dataset_path, negative_dataset_path, coord_exchange = False):
        self.name = dataset + '_' + image_set
        self.image_set = image_set
        self.root_path = root_path
        self.positive_data_path = positive_dataset_path
        self.negative_data_path = negative_dataset_path

        self.classes = ['__background__', 'idcard_frontal']
        self.num_classes = 2
        self.positive_image_set_index = self.load_image_set_index(self.positive_data_path)
        self.negative_image_set_index = self.load_image_set_index(self.negative_data_path)
        self.num_images = len(self.positive_image_set_index) + len(self.negative_image_set_index)
        self.coord_exchange = coord_exchange  # coord_exchange means if the image is upside down, use its left-bottom corner to be top-left corner


    @property
    def cache_path(self):
        """Make a directory to store all caches

        Parameters:
        ----------
        Returns:
        -------
        cache_path: str
            directory to store caches
        """
        cache_path = os.path.join(self.root_path, 'cache')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path


    def load_image_set_index(self, data_path):
        """Get image index

        Parameters:
        ----------
        Returns:
        -------
        image_set_index: str
            relative path of image
        """
        image_set_index_file = os.path.join(data_path, 'imglists', self.image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file, 'r') as f:
            image_set_index = [x.strip().split(' ')[0] for x in f.readlines()]
        return image_set_index


    def gt_imdb(self):
        """Get and save ground truth image database

        Parameters:
        ----------
        Returns:
        -------
        gt_imdb: dict
            image database with annotations
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                imdb = cPickle.load(f)
            print '{} gt imdb loaded from {}'.format(self.name, cache_file)
            return imdb
        positive_gt_imdb = self.load_annotations(positive = True)
        negative_gt_imdb = self.load_annotations(positive = False)
        with open(cache_file, 'wb') as f:
            cPickle.dump([positive_gt_imdb, negative_gt_imdb], f, cPickle.HIGHEST_PROTOCOL)
        return positive_gt_imdb, negative_gt_imdb


    def image_path_from_index(self, index, positive = True):
        """Given image index, return full path

        Parameters:
        ----------
        index: str
            relative path of image
        Returns:
        -------
        image_file: str
            full path of image
        """
        """
        if positive:
            image_file = os.path.join(self.positive_data_path, 'images', index)
        else:
            image_file = os.path.join(self.negative_data_path, 'images', index)
        """
        image_file = index
        if "." not in image_file:
            image_file = image_file + '.JPEG'
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file


    def load_annotations(self, positive = True):
        def coordinates_exchange(coordinates):
            x_axises = coordinates[0:8:2]
            y_axises = coordinates[1:8:2]
            mul = []
            for idx, x_axis in enumerate(x_axises):
                mul.append(x_axis * y_axises[idx])
            sorted_idx = sorted(range(len(mul)), key=lambda k: mul[k])
            sorted_idx = sorted_idx[0] # get smallest product, which is the top-left coorner
            coordinates_new = coordinates[2 * sorted_idx: 2* sorted_idx + 2]
            idx = sorted_idx + 1
            while idx != sorted_idx:
                if idx > 4:
                    idx = 0
                coordinates_new.append(coordinates[2 * idx: 2* idx + 2])
            print coordinates_new
            return coordinates_new


        """Load annotations

        Parameters:
        ----------
        Returns:
        -------
        imdb: dict
            image database with annotations
        """
        if positive:
            annotation_file = os.path.join(self.positive_data_path, 'imglists', self.image_set + '.txt')
            num_images = len(self.positive_image_set_index)
        else:
            annotation_file = os.path.join(self.negative_data_path, 'imglists', self.image_set + '.txt')
            num_images = len(self.negative_image_set_index)
        assert os.path.exists(annotation_file), 'annotations not found at {}'.format(annotation_file)
        with open(annotation_file, 'r') as f:
            annotations = f.readlines()

        imdb = []
        for i in range(num_images):
            annotation = annotations[i].strip().split(' ')
            index = annotation[0]
            im_path = self.image_path_from_index(index, positive = positive)
            imdb_ = dict()
            imdb_['image'] = im_path
            if self.image_set == 'test':
                pass
            else:
                label = annotation[1]
                imdb_['label'] = int(label)
                imdb_['flipped'] = False
                imdb_['bbox_target'] = np.zeros((8,))
                if len(annotation[2:]) == 8:
                    bbox_target = annotation[2:]
                    if self.coord_exchange:
                        bbox_target = coordinates_exchange(bbox_target)
                    imdb_['bbox_target'] = np.array(bbox_target).astype(float)

            imdb.append(imdb_)
        return imdb



    def append_flipped_images(self, imdb):
        """append flipped images to imdb

        Parameters:
        ----------
        imdb: imdb
            image database
        Returns:
        -------
        imdb: dict
            image database with flipped image annotations added
        """
        print 'append flipped images to imdb', len(imdb)
        for i in range(len(imdb)):
            imdb_ = imdb[i]
            m_bbox = imdb_['bbox_target'].copy()
            m_bbox[0], m_bbox[2] = -m_bbox[2], -m_bbox[0]

            entry = {'image': imdb_['image'],
                     'label': imdb_['label'],
                     'bbox_target': m_bbox,
                     'flipped': True}

            imdb.append(entry)
        self.image_set_index *= 2
        return imdb

    def write_results(self, all_boxes):
        """write results

        Parameters:
        ----------
        all_boxes: list of numpy.ndarray
            detection results
        Returns:
        -------
        """
        print 'Writing fddb results'
        res_folder = os.path.join(self.cache_path, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)

        # save results to fddb format
        filename = os.path.join(res_folder, self.image_set + '-out.txt')
        with open(filename, 'w') as f:
            for im_ind, index in enumerate(self.image_set_index):
                f.write('%s\n'%index)
                dets = all_boxes[im_ind]
                f.write('%d\n'%dets.shape[0])
                if len(dets) == 0:
                    continue
                for k in range(dets.shape[0]):
                    f.write('{:.2f} {:.2f} {:.2f} {:.2f} {:.5f}\n'.
                            format(dets[k, 0], dets[k, 1], dets[k, 2]-dets[k, 0], dets[k, 3]-dets[k, 1], dets[k, 4]))
