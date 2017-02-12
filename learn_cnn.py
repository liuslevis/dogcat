# https://www.tensorflow.org/tutorials/mnist/beginners/
from scipy import ndimage,misc
import numpy as np
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# TRAIN_PATH = 'data/input/64/train_small'
# TEST_PATH  = 'data/input/64/test_small'
TRAIN_PATH = 'data/input/64/train'
TEST_PATH  = 'data/input/64/test'
CATES = ['dog', 'cat']
NUM_LABELS = len(CATES)
IMG_W = 64
IMG_H = 64
IMG_SIZE = IMG_W * IMG_H
IMG_SUFFIX = 'jpg'
BATCH_SIZE = 20 # num of img each train iter
DEBUG = True

def read_images_labels(path, part_range, categories = CATES):
    images = []
    labels = []
    files = []
    for file in os.listdir(path):
        if IMG_SUFFIX in file:
            if path == TRAIN_PATH:
                category, index, suffix = file.split('.')                    
                label = categories.index(category)
                if int(index) not in part_range:
                    continue
            elif path == TEST_PATH:
                index, suffix = file.split('.')
                if int(index) not in part_range:
                    continue
            pic = misc.imread(os.path.join(path, file))
            img = pic[:,:,0] # gray channel
            flat_img = img.reshape(img.shape[0]*img.shape[1])
            images.append(np.float32(flat_img/128.0))
            labels.append(label)
            files.append(file)
    return np.array(images), np.eye(len(categories))[labels]

def num_images(path):
    return len(list(filter(lambda file:file.endswith(IMG_SUFFIX), os.listdir(path))))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1], padding='SAME')

# n pieces of 64*64 imgs
x = tf.placeholder(tf.float32, [None, IMG_SIZE])
x_image = tf.reshape(x, [-1,64,64,1])

# conv 1 -> img 64*64 * 32
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# maxpool 1, take max value of 2x2 area, -> img 32*32 * 32
h_pool1 = max_pool_2x2(h_conv1)

# conv2 -> img 32*32 * 64
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# maxpool 2 -> img 16*16 * 64
h_pool2 = max_pool_2x2(h_conv2)

# dense connected area -> 2
W_fc1 = weight_variable([16 * 16 * 64, 2])
b_fc1 = bias_variable([2])
h_pool1_flat = tf.reshape(h_pool2, [-1, 16 * 16 * 64])
y = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

# ground truth
y_ = tf.placeholder(tf.float32, [None, NUM_LABELS]) # n * NUM_LABELS

# cost
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

#train
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

def train_test_ranges(train_proportion):
    iters = int(num_images(TRAIN_PATH) / BATCH_SIZE / NUM_LABELS)

    ranges = [range(i * BATCH_SIZE, (i+1) * BATCH_SIZE) for i in range(0, iters)]
    TRAIN_PART = 0.7
    split_index = int(len(ranges) * TRAIN_PART)
    train_ranges = ranges[:split_index]
    test_ranes = ranges[split_index:]
    return train_ranges, test_ranes

train_ranges, test_ranges = train_test_ranges(0.7)

for rang in train_ranges:
    step = train_ranges.index(rang)
    batch_xs, batch_ys = read_images_labels(TRAIN_PATH, rang)
    if step % 1 == 0:
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('step:%d range:%s accuracy:%.4f' % (step, rang, sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})))
        # if DEBUG: print('learned: \nb:\n%s \nW:\n%s \ny:\n%s' % (sess.run(b), sess.run(W), sess.run(y, feed_dict={x: batch_xs})))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
# Evaluation

for rang in test_ranges[:1]:
    batch_xs, batch_ys = read_images_labels(TRAIN_PATH, rang)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('test accuracy:%.4f' % sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))
