"""Functions for downloading and reading MNIST data."""
import gzip
import os
import urllib
import numpy
from pdb import set_trace as st
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'


def maybe_download(filename, work_directory):
    """Download the data from Yann's website, unless it's already here."""
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(int(rows * cols * num_images))
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(int(num_images), int(rows), int(cols), 1)
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename, one_hot=False):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(int(num_items))
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        if one_hot:
            return dense_to_one_hot(labels)
        return labels


class DataSet(object):
    def __init__(self, images, labels, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], (
                "images.shape: %s labels.shape: %s" % (images.shape,
                                                       labels.shape))
            self._num_examples = images.shape[0]
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(numpy.float32)
            images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1.0 for _ in xrange(784)]
            fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir, fake_data=False, one_hot=False):
    class DataSets(object):
        pass
    data_sets = DataSets()
    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True)
        data_sets.validation = DataSet([], [], fake_data=True)
        data_sets.test = DataSet([], [], fake_data=True)
        return data_sets
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
    VALIDATION_SIZE = 5000
    local_file = maybe_download(TRAIN_IMAGES, train_dir)
    train_images = extract_images(local_file)
    local_file = maybe_download(TRAIN_LABELS, train_dir)
    train_labels = extract_labels(local_file, one_hot=one_hot)
    local_file = maybe_download(TEST_IMAGES, train_dir)
    test_images = extract_images(local_file)
    local_file = maybe_download(TEST_LABELS, train_dir)
    test_labels = extract_labels(local_file, one_hot=one_hot)
    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    # unbalanced data for two classes 0, 1
    class0_count = 0
    class1_count = 0
    add_idx_unbalance01 = []
    unbalance01_labels = []
    for idx, label in enumerate(train_labels):
        if (label == numpy.array([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])).all():
        # if (label == numpy.array([ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.])).all():
            # add_idx_unbalance01.append(idx)
            class0_count += 1
            if class0_count <= 4545:
                add_idx_unbalance01.append(idx)
                unbalance01_labels.append(numpy.array([ 1.,  0.]))
        if (label == numpy.array([ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])).all():
        # if (label == numpy.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.])).all():
            class1_count += 1
            if class1_count <= 455:
                add_idx_unbalance01.append(idx)
                unbalance01_labels.append(numpy.array([ 0.,  1.]))

        if class0_count > 4545 and class1_count > 455:
            break

    unbalance01_images = train_images[add_idx_unbalance01]
    unbalance01_labels = numpy.array(unbalance01_labels)
    data_sets.unbalance01 = DataSet(unbalance01_images, unbalance01_labels)

    # balance data for two classes 0, 1
    # training
    class0_count = 0
    class1_count = 0
    add_idx_balance01 = []
    balance01_labels = []
    for idx, label in enumerate(train_labels):
        if (label == numpy.array([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])).all():
            class0_count += 1
            if class0_count <= 5000:
                add_idx_balance01.append(idx)
                balance01_labels.append(numpy.array([ 1.,  0.]))
        if (label == numpy.array([ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])).all():
            class1_count += 1
            if class1_count <= 5000:
                add_idx_balance01.append(idx)
                balance01_labels.append(numpy.array([ 0.,  1.]))
        if class0_count > 5000 and class1_count > 5000:
            break

    balance01_images = train_images[add_idx_balance01]
    balance01_labels = numpy.array(balance01_labels)
    data_sets.balance01 = DataSet(balance01_images, balance01_labels)

    # test
    class0_count = 0
    class1_count = 0
    add_idx_balance01_test = []
    balance01_test_labels = []
    for idx, label in enumerate(test_labels):
        if (label == numpy.array([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])).all():
            class0_count += 1
            if class0_count <= 500:
                add_idx_balance01_test.append(idx)
                balance01_test_labels.append(numpy.array([ 1.,  0.]))
        if (label == numpy.array([ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])).all():
            class1_count += 1
            if class1_count <= 500:
                add_idx_balance01_test.append(idx)
                balance01_test_labels.append(numpy.array([ 0.,  1.]))

        if class0_count > 500 and class1_count > 500:
            break

    balance01_test_images = test_images[add_idx_balance01_test]
    balance01_test_labels = numpy.array(balance01_test_labels)
    data_sets.balance01_test = DataSet(balance01_test_images, balance01_test_labels)

    data_sets.train = DataSet(train_images, train_labels)
    data_sets.validation = DataSet(validation_images, validation_labels)
    data_sets.test = DataSet(test_images, test_labels)
    
    return data_sets
