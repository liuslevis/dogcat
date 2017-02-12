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
IMG_SIZE = 64 * 64
IMG_SUFFIX = 'jpg'
BATCH_SIZE = 5 # num of img each train iter
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
            images.append(np.float32(flat_img)/128.0)
            labels.append(label)
            files.append(file)
    return np.array(images), np.eye(len(categories))[labels]

def num_images(path):
    return len(list(filter(lambda file:file.endswith(IMG_SUFFIX), os.listdir(path))))

# n pieces of 64 * 64 imgs
x = tf.placeholder(tf.float32, [None, IMG_SIZE])

W = tf.Variable(tf.zeros([IMG_SIZE, NUM_LABELS]))

b = tf.Variable(tf.zeros([NUM_LABELS]))

y = tf.nn.softmax(tf.matmul(x, W) + b) # n * NUM_LABELS

y_ = tf.placeholder(tf.float32, [None, NUM_LABELS]) # n * NUM_LABELS

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y))) # reduce_sum accum second dim of its param

train_step = tf.train.GradientDescentOptimizer(learning_rate=.00001).minimize(cross_entropy)

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
    if step % 20 == 1:
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('step:%d range:%s accuracy:%.4f' % (step, rang, sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})))
        if DEBUG: print('learned: \nb:\n%s \nW:\n%s \ny:\n%s' % (sess.run(b), sess.run(W), sess.run(y, feed_dict={x: batch_xs})))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
# Evaluation

for rang in test_ranges[:1]:
    batch_xs, batch_ys = read_images_labels(TRAIN_PATH, rang)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('test accuracy:%.4f' % sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))
