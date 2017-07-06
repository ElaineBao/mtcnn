import mxnet as mx
from mxnet.executor_manager import _split_input_slice
import numpy as np
import minibatch
from config import config

class TestLoader(mx.io.DataIter):
    def __init__(self, imdb, batch_size=1, shuffle=False):
        self.imdb = imdb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size = len(imdb)
        self.index = np.arange(self.size)

        self.cur = 0
        self.data = None
        self.label = None

        self.data_names = ['data']
        self.label_names = []

        self.reset()
        self.get_batch()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self.label_names, self.label)]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        imdb = [self.imdb[self.index[i]] for i in range(cur_from, cur_to)]
        data, label = minibatch.get_testbatch(imdb)
        self.data = [mx.nd.array(data[name]) for name in self.data_names]
        self.label = [mx.nd.array(label[name]) for name in self.label_names]

class ImageLoader(mx.io.DataIter):
    def __init__(self, imdb_positive, imdb_negative, im_size, batch_size=config.BATCH_SIZE, shuffle=False, ctx=None, work_load_list=None):

        super(ImageLoader, self).__init__()

        self.imdb_positive = imdb_positive
        self.imdb_negative = imdb_negative
        self.batch_size = batch_size
        self.batch_size_positive = int(batch_size * config.BATCH_FG_FRACTION)
        self.batch_size_negative = batch_size - self.batch_size_positive
        self.im_size = im_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list

        self.cur_positive = 0
        self.size_positive = len(imdb_positive)
        self.cur_negative = 0
        self.size_negative = len(imdb_negative)

        self.index_positive = np.arange(self.size_positive)
        self.index_negative = np.arange(self.size_negative)
        self.num_classes = 2

        self.batch = None
        self.data = None
        self.label = None

        self.label_names= ['label', 'bbox_target']
        self.reset()
        self.get_batch()

    @property
    def provide_data(self):
        return [('data', self.data[0].shape)]
      #  return [(k, v.shape) for k, v in zip(self.data_name, self.data)]


    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self.label_names, self.label)]


    def reset(self):
        self.cur_positive = 0
        self.cur_negative = 0
        if self.shuffle:
            np.random.shuffle(self.index_positive)
            np.random.shuffle(self.index_negative)


    def iter_next(self):
        return (self.cur_positive + self.batch_size_positive <= self.size_positive) \
               and (self.cur_negative + self.batch_size_negative <= self.size_negative)

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur_positive += self.batch_size_positive
            self.cur_negative += self.batch_size_negative
            return mx.io.DataBatch(data=self.data, label=self.label, #pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    """
    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0
    """

    def get_batch(self):

        def tensor_vstack(tensor_list, pad=0):
            """
            vertically stack tensors
            :param tensor_list: list of tensor to be stacked vertically
            :param pad: label to pad with
            :return: tensor with max shape
            """
            ndim = len(tensor_list[0].shape)
            dtype = tensor_list[0].dtype
            islice = tensor_list[0].shape[0]
            dimensions = []
            first_dim = sum([tensor.shape[0] for tensor in tensor_list])
            dimensions.append(first_dim)
            for dim in range(1, ndim):
                dimensions.append(max([tensor.shape[dim] for tensor in tensor_list]))
            if pad == 0:
                all_tensor = np.zeros(tuple(dimensions), dtype=dtype)
            elif pad == 1:
                all_tensor = np.ones(tuple(dimensions), dtype=dtype)
            else:
                all_tensor = np.full(tuple(dimensions), pad, dtype=dtype)
            if ndim == 1:
                for ind, tensor in enumerate(tensor_list):
                    all_tensor[ind * islice:(ind + 1) * islice] = tensor
            elif ndim == 2:
                for ind, tensor in enumerate(tensor_list):
                    all_tensor[ind * islice:(ind + 1) * islice, :tensor.shape[1]] = tensor
            elif ndim == 3:
                for ind, tensor in enumerate(tensor_list):
                    all_tensor[ind * islice:(ind + 1) * islice, :tensor.shape[1], :tensor.shape[2]] = tensor
            elif ndim == 4:
                for ind, tensor in enumerate(tensor_list):
                    all_tensor[ind * islice:(ind + 1) * islice, :tensor.shape[1], :tensor.shape[2],
                    :tensor.shape[3]] = tensor
            else:
                raise Exception('Sorry, unimplemented.')
            return all_tensor

        positive_cur_from = self.cur_positive
        positive_cur_to = min(positive_cur_from + self.batch_size_positive, self.size_positive)
        imdb = [self.imdb_positive[self.index_positive[i]] for i in range(positive_cur_from, positive_cur_to)]
        negative_cur_from = self.cur_negative
        negative_cur_to = min(negative_cur_from + self.batch_size_negative, self.size_negative)
        for i in range(negative_cur_from, negative_cur_to):
            imdb.append(self.imdb_negative[self.index_negative[i]])
        np.random.shuffle(imdb)

        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        # get testing data for multigpu
        # each element in the list is the data used by different gpu
        data_list = []
        label_list = []
        for islice in slices:
            i_imdb = [imdb[i] for i in range(islice.start, islice.stop)]
            data, label = minibatch.get_minibatch(i_imdb, self.num_classes, self.im_size)
            data_list.append(data)
            label_list.append(label)

        all_data = dict()
        for key in data_list[0].keys():
            all_data[key] = tensor_vstack([batch[key] for batch in data_list])

        all_label = dict()
        for key in label_list[0].keys():
            all_label[key] = tensor_vstack([batch[key] for batch in label_list])

        self.data = [mx.nd.array(all_data['data'])]
        self.label = [mx.nd.array(all_label[name]) for name in self.label_name]


