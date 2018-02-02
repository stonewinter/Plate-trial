import tensorflow as tf
import numpy as np
import cv2
import opencvlib

def GetAccuracy(session, featureSet, labelSet, predictFunc):
    global Xholder, Yholder;
    y_pre = session.run(predictFunc, feed_dict={Xholder: featureSet})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(labelSet, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # # 拿真实的数据在最新的更新好参数的网络上进行一次准确率测试
    result = session.run(accuracy,
                         feed_dict={Xholder: featureSet, Yholder: labelSet}
                         )
    return result;




featureSet = np.load("features.npy");
labelSet = np.load("labels.npy");
W = tf.Variable(tf.zeros([featureSet.shape[1], labelSet.shape[1]]), dtype=tf.float32)
b = tf.Variable(tf.zeros([1, labelSet.shape[1]]), dtype=tf.float32)

sess = tf.Session();
tf.train.Saver().restore(sess, "./PlateRecogNet/PlateRecogNet.ckpt")

Xholder = tf.placeholder(tf.float32, [None, featureSet.shape[1]])
Yholder = tf.placeholder(tf.float32, [None, labelSet.shape[1]])
predict_outputs = tf.nn.softmax(tf.matmul(Xholder, W) + b)
print("acc=",GetAccuracy(sess,featureSet,labelSet,predict_outputs))

img = cv2.imread("./chepai_test2.jpg");
resize = opencvlib.Resize(img, 60, 20);
X = np.array([resize.flatten() / 255]);

prediction = sess.run(predict_outputs, feed_dict={Xholder: X});
print("prediction", prediction)
