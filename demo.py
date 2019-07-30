import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
import os
import re
import pickle
import argparse
import tensorflow as tf
import net
import csv

ap = argparse.ArgumentParser()
ap.add_argument("--image",required=True,
               help='Path of the human face image')
ap.add_argument("--usecamera",required = True,
               help='1 means using camera,0 means not using camera')
args=vars(ap.parse_args())

def load_model():
    sess=tf.Session()
    x = tf.placeholder(tf.float32,[None,48,48,1])
    y_smile_conv,y_gender_conv,y_glasses_conv,phase_train,keep_prob=net.BKNetModel(x)
    saver = tf.train.Saver(max_to_keep=1)
    print('Restoring existed model')
    saver.restore(sess, './save/current/model.ckpt')
    print('OK')

    return sess,x,y_smile_conv,y_gender_conv,y_glasses_conv,phase_train,keep_prob

def draw_label(img,x,y,w,h,label,font=cv2.FONT_HERSHEY_SIMPLEX,font_scale=1,thickness=2):
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,155,255),2)
    cv2.putText(img,label,(x,y),font,font_scale,(255,255,255),thickness)

def main(sess,x,y_smile_conv,y_gender_conv,y_glasses_conv,phase_train,keep_prob):
    detector = MTCNN()
    if(int(args['usecamera'])==1):
        cap = cv2.VideoCapture(0)
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                print("error: failed to capture image")
                return -1

            # detect face and crop face, convert to gray, resize to 48x48
            original_img = img
            cv2.imshow("result", original_img)
            result = detector.detect_faces(original_img)
            if not result:
                cv2.imshow("result", original_img)
                continue
            face_position = result[0].get('box')
            x_coordinate = face_position[0]
            y_coordinate = face_position[1]
            w_coordinate = face_position[2]
            h_coordinate = face_position[3]
            img = original_img[y_coordinate:y_coordinate + h_coordinate, x_coordinate:x_coordinate + w_coordinate]
            if (img.size == 0):
                cv2.imshow("result", original_img)
                continue;
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (48, 48))
            img = (img - 128) / 255.0
            T = np.zeros([48, 48, 1])
            T[:, :, 0] = img
            test_img = []
            test_img.append(T)
            test_img = np.asarray(test_img)

            predict_y_smile_conv = sess.run(y_smile_conv, feed_dict={x: test_img, phase_train: False, keep_prob: 1})
            predict_y_gender_conv = sess.run(y_gender_conv, feed_dict={x: test_img, phase_train: False, keep_prob: 1})
            predict_y_glasses_conv = sess.run(y_glasses_conv, feed_dict={x: test_img, phase_train: False, keep_prob: 1})

            smile_label = "-_-" if np.argmax(predict_y_smile_conv) == 0 else ":)"
            gender_label = "Female" if np.argmax(predict_y_gender_conv) == 0 else "Male"
            glasses_label = 'On Glasses' if np.argmax(predict_y_glasses_conv)==1 else 'No Glasses'

            label = "{}, {}, {}".format(smile_label, gender_label, glasses_label)
            draw_label(original_img, x_coordinate, y_coordinate, w_coordinate, h_coordinate, label)

            cv2.imshow("result", original_img)
            key = cv2.waitKey(1)

            if key == 27:
                break

    else:
        img_list = os.listdir(args['image'])
        with open('label.csv','a') as csv_file:
            writer = csv.writer(csv_file,delimiter = ',')
            for img_name in img_list:
                label_list = []
                original_img = cv2.imread(os.path.join(args['image'],img_name))
                result = detector.detect_faces(original_img)
                if not result:
                    print('can not detect face in the photo')
                    print(img_name)
                    continue
                face_position = result[0].get('box')
                x_coordinate = face_position[0]
                y_coordinate = face_position[1]
                w_coordinate = face_position[2]
                h_coordinate = face_position[3]
                img = original_img[y_coordinate:y_coordinate + h_coordinate, x_coordinate:x_coordinate + w_coordinate]
                if img.size ==0:
                    print('can not crop the face from the photo')
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (48, 48))
                img = (img - 128) / 255.0
                T = np.zeros([48, 48, 1])
                T[:, :, 0] = img
                test_img = []
                test_img.append(T)
                test_img = np.asarray(test_img)

                predict_y_smile_conv = sess.run(y_smile_conv, feed_dict={x: test_img, phase_train: False, keep_prob: 1})
                predict_y_gender_conv = sess.run(y_gender_conv, feed_dict={x: test_img, phase_train: False, keep_prob: 1})
                predict_y_glasses_conv = sess.run(y_glasses_conv, feed_dict={x: test_img, phase_train: False, keep_prob: 1})

                label_list.append(img_name)
                label_list.append( '-_-' if np.argmax(predict_y_smile_conv)==0 else ':)')
                label_list.append('Female' if np.argmax(predict_y_gender_conv)==0 else 'Male')
                label_list.append('On Glasses' if np.argmax(predict_y_glasses_conv)==1 else 'No Glasses')
                writer.writerow(label_list)

                label = "{}, {}, {}".format(label_list[1], label_list[2], label_list[3])
                draw_label(original_img, x_coordinate, y_coordinate, w_coordinate, h_coordinate, label)

                cv2.imshow("result", original_img)
                key = cv2.waitKey(1)

if __name__ == '__main__':
    sess, x, y_smile_conv, y_gender_conv, y_glasses_conv, phase_train, keep_prob = load_model()
    main(sess,x,y_smile_conv,y_gender_conv,y_glasses_conv,phase_train,keep_prob)










