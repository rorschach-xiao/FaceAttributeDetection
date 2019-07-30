import CNNhead_input
import tensorflow as tf
import numpy as np
import net
from const import *

NUMBER_SMILE_TEST = 1000
NUMBER_GENDER_TEST = 1118

''' PREPARE DATA '''
smile_train, smile_test = CNNhead_input.getSmileImage()
gender_train, gender_test = CNNhead_input.getGenderImage()
glasses_train, glasses_test = CNNhead_input.getGlassesImage()

def one_hot(index, num_classes):
    assert index ==-1 or index == 1
    tmp = np.zeros(num_classes, dtype=np.float32)
    if index ==-1:
        tmp[0] = 1.0
    else:
        tmp[1]=1.0
    return tmp

with tf.Session() as sess:
    x, y_, mask = net.Input()
    y_smile_conv, y_gender_conv, y_glasses_conv, phase_train, keep_prob = net.BKNetModel(x)

    smile_loss, gender_loss, glasses_loss, l2_loss, loss = net.selective_loss(y_smile_conv, y_gender_conv,
                                                                                 y_glasses_conv, y_, mask)
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")


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


    test_data = []

    # Mask: Smile -> 0, Gender -> 1, Glasses -> 2
    for i in range(len(smile_test)):
        img = (smile_test[i][0] - 128) / 255.0
        label = (int)(smile_test[i][1])
        test_data.append((img, one_hot(label, 2), 0.0))
    for i in range(len(gender_test)):
        img = (gender_test[i][0] - 128) / 255.0
        label = (int)(gender_test[i][1])
        test_data.append((img, one_hot(label, 2), 1.0))
    for i in range(len(glasses_test)):
        img = (glasses_test[i][0] - 128) / 255.0
        label = (int)(glasses_test[i][1])
        test_data.append((img, one_hot(label, 2), 2.0))
    np.random.shuffle(test_data)

    saver = tf.train.Saver(max_to_keep=1)
    print('Restoring existed model')
    saver.restore(sess, './save/current/model.ckpt')
    print('OK')

    test_img = []
    test_label = []
    test_mask = []

    for i in range(len(test_data)):
        test_img.append(test_data[i][0])
        test_label.append(test_data[i][1])
        test_mask.append(test_data[i][2])

    number_batch = len(test_data) // BATCH_SIZE

    smile_nb_true_pred = 0
    gender_nb_true_pred = 0
    glasses_nb_true_pred = 0

    smile_nb_test = 0
    gender_nb_test = 0
    glasses_nb_test = 0
    print("length of test data :"+str(len(test_data)))
#     for batch in range(number_batch):
    for batch in range(number_batch):
        top = batch * BATCH_SIZE
        bot = min((batch + 1) * BATCH_SIZE, len(test_data))
        batch_img = np.asarray(test_img[top:bot])
        batch_label = np.asarray(test_label[top:bot])
        batch_mask = np.asarray(test_mask[top:bot])
        for i in range(BATCH_SIZE):
            if batch_mask[i] == 0.0:
                    smile_nb_test += 1
            else:
                if batch_mask[i] == 1.0:
                    gender_nb_test += 1
                else:
                    glasses_nb_test += 1

        batch_img = np.reshape(batch_img, (BATCH_SIZE, 48, 48, 1))

        smile_nb_true_pred += sess.run(smile_true_pred, feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                                   phase_train: False,
                                                                   keep_prob: 1})

        gender_nb_true_pred += sess.run(gender_true_pred,
                                        feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                   phase_train: False,
                                                   keep_prob: 1})

        glasses_nb_true_pred += sess.run(glasses_true_pred,
                                         feed_dict={x: batch_img, y_: batch_label, mask: batch_mask,
                                                    phase_train: False,
                                                    keep_prob: 1})
        print("batch "+str(batch)+" smile tp :"+str(smile_nb_true_pred)+ " gender tp :"+str(gender_nb_true_pred) +" glasses tp :"+str(glasses_nb_true_pred))
    print("smile test number :" + str(smile_nb_test))
    print("gender test number :" + str(gender_nb_test))
    print("glasses test number :" + str(glasses_nb_test))
    smile_test_accuracy = smile_nb_true_pred * 1.0 / smile_nb_test
    gender_test_accuracy = gender_nb_true_pred * 1.0 / gender_nb_test
    glasses_test_accuracy = glasses_nb_true_pred * 1.0 / glasses_nb_test


    print('Smile task test accuracy: ' + str(smile_test_accuracy * 100))
    print('Gender task tst accuracy: ' + str(gender_test_accuracy * 100))
    print('Age task test accuracy: ' + str(glasses_test_accuracy * 100))

    print('\n')

