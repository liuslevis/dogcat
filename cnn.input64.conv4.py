# https://www.tensorflow.org/tutorials/mnist/beginners/
from scipy import ndimage,misc
import numpy as np
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

IMG_H = IMG_W = 64
IMG_SIZE = IMG_W * IMG_H

# TRAIN_PATH = 'data/input/64/train_small'
# TEST_PATH  = 'data/input/64/test_small'
TRAIN_PATH = 'data/input/%s/train' % IMG_H
TEST_PATH  = 'data/input/%s/test'  % IMG_H
CATES = ['dog', 'cat']
NUM_LABELS = len(CATES)
IMG_SUFFIX = 'jpg'
BATCH_SIZE = 50 # num of img each train iter
OUTPUT_STEP = 1
EPOCH = 10
DEBUG = True
TRAIN_PART = 0.7


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
            images.append(np.float32(flat_img)/128.0)
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

# n pieces of imgs
x = tf.placeholder(tf.float32, [None, IMG_SIZE])

x_image = tf.reshape(x, [-1,IMG_W,IMG_H,1])

# conv 1 -> img * 32(channel)
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# maxpool 1, take max value of 2x2 area, -> img/2 * 32(channel)
h_pool1 = max_pool_2x2(h_conv1)

# covn2 -> img/2 * 64(channel)
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# maxpool2 -> img/4 * 64(channel)
h_pool2 = max_pool_2x2(h_conv2)

# conv3 -> img/4 * 128(channel)
W_conv3 = weight_variable([5,5,64,128])
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

# maxpool3 -> img/8 * 128(channel)
# h_pool3 = max_pool_2x2(h_conv3)
h_pool3 = tf.reshape(h_conv3, [-1, 16, 16, 128])

# conv4 
W_conv4 = weight_variable([5,5,128,128])
b_conv4 = bias_variable([128])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)

# maxpool4 -> img/8 * 128(channel)
h_pool4 = max_pool_2x2(h_conv4)

# fc1 -> 256
scale = 8 * 8
W_fc1 = weight_variable([int(IMG_SIZE/scale)*128, 256])
b_fc1 = bias_variable([256])
h_pool4_flat = tf.reshape(h_pool4, [-1, int(IMG_SIZE/scale)*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

# dropout avoid overfitting
# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# fc2
W_fc2 = weight_variable([256, 64])
b_fc2 = bias_variable([64])
h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

#fc3
W_fc3 = weight_variable([64, 2])
b_fc3 = bias_variable([2])
h_fc3 = tf.matmul(h_fc2, W_fc3) + b_fc3

#readout
y = tf.nn.softmax(h_fc3)

y_ = tf.placeholder(tf.float32, [None, NUM_LABELS]) # n * NUM_LABELS

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y))) # reduce_sum accum second dim of its param

# train_step = tf.train.GradientDescentOptimizer(learning_rate=.00001).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

def train_test_ranges(train_proportion):
    iters = int(num_images(TRAIN_PATH) / BATCH_SIZE / NUM_LABELS)
    ranges = [range(i * BATCH_SIZE, (i+1) * BATCH_SIZE) for i in range(iters)]
    split_index = int(len(ranges) * TRAIN_PART)
    train_ranges = ranges[:split_index]
    test_ranes = ranges[split_index:]
    return train_ranges, test_ranes

train_ranges, test_ranges = train_test_ranges(0.7)

for epoch in  range(EPOCH):
    print('\nepoch %d' % epoch)
    for rang in train_ranges:
        step = train_ranges.index(rang)
        batch_xs, batch_ys = read_images_labels(TRAIN_PATH, rang)
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        acc = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
        
        if step % OUTPUT_STEP == 0:
            print('step:%d %s accuracy:%.4f' % (step, rang, acc))
        
        if DEBUG: 
            out = [
                sess.run(h_pool3, feed_dict={x: batch_xs, y_: batch_ys}),
                sess.run(h_conv4, feed_dict={x: batch_xs, y_: batch_ys}),
                sess.run(W_conv4, feed_dict={x: batch_xs, y_: batch_ys}),
            ]
            # print('learned:', out[2].shape, out[2])
         
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Evaluation

for rang in test_ranges[:1]:
    batch_xs, batch_ys = read_images_labels(TRAIN_PATH, rang)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('test accuracy:%.4f' % sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))
