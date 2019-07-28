import CNNhead_input as CNN2Head_input
import os
import tensorflow as tf
import numpy as np
import net
from const import *

''' PREPARE DATA '''
''' PREPARE DATA '''
smile_train, smile_test = CNN2Head_input.getSmileImage()
gender_train, gender_test = CNN2Head_input.getGenderImage()
glasses_train, glasses_test = CNN2Head_input.getGlassesImage()


def one_hot(index, num_classes):
    assert index ==-1 or index == 1
    tmp = np.zeros(num_classes, dtype=np.float32)
    if index ==-1:
        tmp[0] = 1.0
    else:
        tmp[1]=1.0
    return tmp

sess = tf.Session()
global_step = tf.contrib.framework.get_or_create_global_step()
x, y_, mask = net.Input()
y_smile_conv, y_gender_conv, y_glasses_conv, phase_train, keep_prob = net.BKNetModel(x)

smile_loss, gender_loss, glasses_loss, l2_loss, loss = net.selective_loss(y_smile_conv, y_gender_conv,
                                                                             y_glasses_conv, y_, mask)
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

train_step = net.train_op(loss, global_step)

smile_mask = tf.get_collection('smile_mask')[0]
gender_mask = tf.get_collection('gender_mask')[0]
glasses_mask = tf.get_collection('glasses_mask')[0]

y_smile = tf.get_collection('y_smile')[0]
y_gender = tf.get_collection('y_gender')[0]
y_glasses = tf.get_collection('y_glasses')[0]

smile_correct_prediction = tf.equal(tf.argmax(y_smile_conv, 1), tf.argmax(y_smile, 1))
gender_correct_prediction = tf.equal(tf.argmax(y_gender_conv, 1), tf.argmax(y_gender, 1))
glasses_correct_prediction = tf.equal(tf.argmax(y_glasses_conv, 1), tf.argmax(y_glasses, 1))

smile_true_pred = tf.reduce_sum(tf.cast(smile_correct_prediction, dtype=tf.float32) * smile_mask)
gender_true_pred = tf.reduce_sum(tf.cast(gender_correct_prediction, dtype=tf.float32) * gender_mask)
glasses_true_pred = tf.reduce_sum(tf.cast(glasses_correct_prediction, dtype=tf.float32) * glasses_mask)
# age_mae, update_op = tf.metrics.mean_absolute_error(
#     tf.argmax(y_glasses, 1), tf.argmax(y_glasses_conv, 1), name="age_mae")
running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="age_mae")
running_vars_initializer = tf.variables_initializer(var_list=running_vars)
sess.run(running_vars_initializer)

train_data = []
# Mask: Smile -> 0, Gender -> 1, Glasses -> 2
for i in range(len(smile_train) * 10):
    img = (smile_train[i][0] - 128) / 255.0
    label = smile_train[i][1]
    train_data.append((img, one_hot(label, 2), 0.0))
for i in range(len(gender_train)):
    img = (gender_train[i][0] - 128) / 255.0
    label = (int)(gender_train[i][1])
    train_data.append((img, one_hot(label, 2), 1.0))
for i in range(len(glasses_train)):
    img = (glasses_train[i][0] - 128) / 255.0
    label = (int)(glasses_train[i][1])
    train_data.append((img, one_hot(label, 2), 2.0))

saver = tf.train.Saver(max_to_keep=1)

if not os.path.isfile('./save/current5/model-age101.ckpt.index'):
    print('Create new model')
    sess.run(tf.global_variables_initializer())
    print('OK')
else:
    print('Restoring existed model')
    saver.restore(sess, './save/current5/model-age101.ckpt.index')
    print('OK')

loss_summary_placeholder = tf.placeholder(tf.float32)
tf.summary.scalar('loss', loss_summary_placeholder)
merge_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("./summary/summary5/", graph=tf.get_default_graph())

learning_rate = tf.get_collection('learning_rate')[0]
current_epoch = (int)(global_step.eval() / (len(train_data) // BATCH_SIZE))
for epoch in range(current_epoch + 1, NUM_EPOCHS):
    print('Epoch:', str(epoch))
    np.random.shuffle(train_data)
    train_img = []
    train_label = []
    train_mask = []

    for i in range(len(train_data)):
        train_img.append(train_data[i][0])
        train_label.append(train_data[i][1])
        train_mask.append(train_data[i][2])

    number_batch = len(train_data) // BATCH_SIZE

    avg_ttl = []
    avg_rgl = []
    avg_smile_loss = []
    avg_gender_loss = []
    avg_age_loss = []

    smile_nb_true_pred = 0
    gender_nb_true_pred = 0
    age_nb_true_pred = 0

    smile_nb_train = 0
    gender_nb_train = 0
    age_nb_train = 0

    print("Learning rate: %f" % learning_rate.eval())
#     for batch in range(number_batch):
    for batch in range(number_batch):
#         if (batch+1)%1000==0:
#             print('Training on batch {0}/{1}'.format(str(batch + 1), str(number_batch)))
        top = batch * BATCH_SIZE
        bot = min((batch + 1) * BATCH_SIZE, len(train_data))
        batch_img = np.asarray(train_img[top:bot])
        batch_label = np.asarray(train_label[top:bot])
        batch_mask = np.asarray(train_mask[top:bot])
    for i in range(BATCH_SIZE):
        if batch_mask[i] == 0.0:
                smile_nb_train += 1
        else:
            if batch_mask[i] == 1.0:
                gender_nb_train += 1
            else:
                age_nb_train += 1
        batch_img = CNN2Head_input.augmentation(batch_img, 48)
        ttl, sml, gel, agl, l2l, _ = sess.run([loss, smile_loss, gender_loss, glasses_loss, l2_loss, train_step],
                                              feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                         phase_train: True,
                                                         keep_prob: 0.5})

        smile_nb_true_pred += sess.run(smile_true_pred, feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                                   phase_train: True,
                                                                   keep_prob: 0.5})

        gender_nb_true_pred += sess.run(gender_true_pred,
                                        feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                   phase_train: True,
                                                   keep_prob: 0.5})

#         age_nb_true_pred += sess.run(age_true_pred,
#                                          feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
#                                                     phase_train: True,
#                                                     keep_prob: 0.5})

        sess.run(update_op,
            feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                       phase_train: False,
                       keep_prob: 1})
