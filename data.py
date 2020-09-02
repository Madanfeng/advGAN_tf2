import os
import cv2
import random
import numpy as np
import tensorflow as tf
import time

from config import cfg

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_IDS


# myself dataset
class Dataset:
    # dataset with iter
    # using dataset to attack model or evaluate model.
    # 目前的版本是从 annotation 文件中读取图片路径和标签 以此读取图片
    # batch_images, batch_labels, batch_target
    # 注意：其中图片已经resize,经过归一化处理 0.5~-0.5， 标签不是one-hot编码
    def __init__(self, istrain):
        self.target = cfg.TARGET  # 目标类，本代码统一将不同图片攻击成某一设定类别
        self.annot_path = cfg.TRAIN_ANNOT_PATH if istrain else cfg.TEST_ANNOT_PATH

        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.num_channels = cfg.NUM_CHANNELS
        self.class_name = cfg.CLASS_NAME
        self.class_num = len(self.class_name)

        self.annotations = self.load_annotations()
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))  # 一共需要分为多少批次
        self.batch_count = 0  # 记录已有多少个batch

        if self.target < 0 or self.target >= self.class_num:
            raise KeyError("The target had an error value.")

    def load_annotations(self):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        # np.random.shuffle(annotations)
        return annotations

    def __iter__(self):
        return self

    def __next__(self):

        with tf.device('/cpu:0'):
            batch_images = np.zeros((self.batch_size, self.image_size, self.image_size, self.num_channels),
                                    dtype=np.float32)
            batch_labels = np.ones(self.batch_size, dtype=np.float32)
            batch_target = np.ones(self.batch_size, dtype=np.float32) * self.target
            batch_path   = []

            num = 0  # 记录当前batch中样本的个数
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples: index -= self.num_samples
                    annotation = self.annotations[index]
                    image, label, image_path = self.parse_annotation(annotation)

                    batch_images[num, :, :, :] = image
                    batch_labels[num] *= label

                    if cfg.TARGETED and int(label) == int(self.target):
                        # 在 目标攻击 的前提下，
                        # 如果图片标签和目标类一样，则随机选取一个目标类
                        target_pool = [i for i in range(self.class_num)]
                        target_pool.pop(self.target)
                        batch_target[num] = np.array(random.sample(target_pool, 1))

                    batch_path.append(image_path)

                    num += 1

                self.batch_count += 1
                return batch_images, batch_labels, batch_target, batch_path
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def __len__(self):
        return self.num_batchs

    def parse_annotation(self, annotation):
        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (self.image_size, self.image_size))
        image_resized = image_resized / 255. - .5

        label = int(line[1])
        return image_resized, label, image_path


# tf.data.Dataset, more efficiently
def dataset(istrain):
    annot_path = cfg.TRAIN_ANNOT_PATH if istrain else cfg.TEST_ANNOT_PATH

    images_path = list()
    labs = list()
    targets = np.full(cfg.BATCH_SIZE, cfg.TARGET, dtype=np.int32)

    with open(annot_path, 'r') as f:
        txt = f.readlines()
        for line in txt:
            path, lab = line.split()
            images_path.append(path)
            labs.append(int(lab))

    def preprocess_image(image):
        image = tf.image.decode_jpeg(image, channels=cfg.NUM_CHANNELS)
        image = tf.image.resize(image, [cfg.IMAGE_SIZE, cfg.IMAGE_SIZE])
        image = image / 255.0 - 0.5  # normalize to [-0.5,0.5] range
        return image

    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        return preprocess_image(image)

    path_ds = tf.data.Dataset.from_tensor_slices(images_path)
    target_ds = tf.data.Dataset.from_tensor_slices(targets)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labs, tf.int64))
    image_label_target_path_ds = tf.data.Dataset.zip((image_ds, label_ds, target_ds, path_ds))

    l = len(labs)
    ds = image_label_target_path_ds.shuffle(buffer_size=l)
    ds = ds.repeat()
    ds = ds.batch(cfg.BATCH_SIZE)
    # 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds, l


if __name__ == '__main__':
    # test different dataset time
    # myself Dataset
    t1 = time.time()
    train_data = Dataset(istrain=False)
    t2 = time.time()
    for epoch in range(4):
        i = 0
        for images, labels, targets, paths in train_data:
            if i == 0:
                print(paths[0])
            i += 1
    t3 = time.time()

    # tf.data.Dataset
    t4 = time.time()
    ds, num = dataset(istrain=False)
    steps_per_epoch = int(num / cfg.BATCH_SIZE)
    print(steps_per_epoch)
    t5 = time.time()
    for epoch in range(4):
        j = 0
        it = iter(ds.take(steps_per_epoch + 1))
        next(it)
        for images, labels, targets, paths in it:
            if j == 0:
                print(paths[0])
            j += 1
    t6 = time.time()

    print("myself dataset time:", t3-t2)
    print("All time:", t3-t1)

    print("tf dataset time:", t6-t5)
    print("All time:", t6-t4)
