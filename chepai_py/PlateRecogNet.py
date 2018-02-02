import tensorflow as tf
import numpy as np;
import cv2;
import opencvlib
import os;



class Batch:
    def __init__(self,featureSet, labelSet):
        if(featureSet.shape[0] != labelSet.shape[0]):
            print("featureSet number != labelSet number");
            return None;
        else:
            self.featureSet = featureSet;
            self.labelSet = labelSet;
            self.endIdx = self.featureSet.shape[0]-1;
            self.currIdx = 0;
    def next_batch(self, batchNbr):
        if (batchNbr > self.featureSet.shape[0]):
            print("batchNbr is too large");
            return None;

        # lastIdx = self.currIdx+batchNbr-1;
        # if(lastIdx < self.endIdx):
        #     retFeatureBatch = self.featureSet[self.currIdx:lastIdx+1];
        #     retLabelBatch = self.labelSet[self.currIdx:lastIdx+1];
        #     self.currIdx = lastIdx+1;
        #     return retFeatureBatch,retLabelBatch;
        # elif(lastIdx == self.endIdx):
        #     retFeatureBatch = self.featureSet[self.currIdx:lastIdx+1];
        #     retLabelBatch = self.labelSet[self.currIdx:lastIdx+1];
        #     self.currIdx = 0;
        #     return retFeatureBatch, retLabelBatch;
        # else:
        #     retFeatureBatch = self.featureSet[self.currIdx:self.endIdx+1];
        #     retLabelBatch = self.labelSet[self.currIdx:self.endIdx+1];
        #     self.currIdx = 0;
        #     return retFeatureBatch, retLabelBatch;

        idxes = np.random.randint(0, self.endIdx+1, batchNbr);
        retFeatureBatch = self.featureSet[idxes,:];
        retLabelBatch = self.labelSet[idxes,:];
        return retFeatureBatch, retLabelBatch;




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

def fit(session, featureSet, labelSet, lossFunc, optimizer, learnRate, epoch, batchNbr):
    global Xholder, Yholder;
    #########################
    ####  构建每步训练操作  ####
    #########################
    train_step = optimizer(learnRate).minimize(lossFunc);

    init = tf.global_variables_initializer()
    batchSet_XY = Batch(featureSet, labelSet);

    session.run(init)
    print("W init:\n", sess.run(W))
    print("b init\n", sess.run(b))
    for i in range(epoch):
        batch_xs, batch_ys = batchSet_XY.next_batch(batchNbr);
        # 拿sub batch数据集合 训练一次网络中各参数,此时网络已更新好了一次
        session.run(train_step, feed_dict={Xholder: batch_xs, Yholder: batch_ys})
        # if i % 50 == 0:
        #     print("loss\n",sess.run(lossFunc, feed_dict={Xholder: featureSet, Yholder: labelSet}))





"""""""""""""""""""修改以下输入数据集合，其余不要动"""""""""""""""""""
"""只需修改featureSet和labelSet内容，变量名不要动"""
featureSet = np.load("features.npy");
labelSet = np.load("labels.npy");
"""""""""""""""""""修改以上输入数据集合，其余不要动"""""""""""""""""""




########################################
### 为真实features，labels添加动态占位符 ###
########################################
# X_real,
# 由于真实数据是28*28的图片，展开后784列一行。共N个图片，我们这里直接给出列数，让系统自行根据784来分行
# None表示：任意行数
Xholder = tf.placeholder(tf.float32, [None, featureSet.shape[1]])
# Y_real，
# 由于是0-9十个输出label类别，那么输出就应以10个元素为一组。
# 到底有多少组，看输入有多少行（每行784个元素）。那么我们和上面一样，取None让其自行分行
Yholder = tf.placeholder(tf.float32, [None, labelSet.shape[1]])




"""""""""""""""""""修改以下网络结构，其余不要动"""""""""""""""""""
####################
###  构建网络结构  ###
####################
# 构建权重阵
# 权重阵是隐藏层，只有一层10个神经元，以784做输入，0-9 十个输出作输出向量
W = tf.Variable(tf.random_normal([featureSet.shape[1], labelSet.shape[1]]))
# 构建偏移阵
# 偏移也是一样的, 根据神经元个数，有10个
b = tf.Variable(tf.zeros([1, labelSet.shape[1]]) + 0.1)
# 构建输出运算，这里只有10个输出
# 强调的是这里的y是网络的预测值，并不是原有的真实label
predict_outputs = tf.nn.softmax(tf.matmul(Xholder, W) + b)
# 所以我们的整体网络结构就是
# 784个输入节点，后跟10个神经元，每个神经元
# 的输出直接作为最终10个输出的一个
"""""""""""""""""""修改以上网络结构，其余不要动"""""""""""""""""""





#####################
#### 构建loss评估 ####
#####################
# 构建loss函数，利用交叉熵cross-entropy
loss = -tf.reduce_sum(Yholder* tf.log(tf.clip_by_value( predict_outputs ,1e-10,1.0)) )


with tf.Session() as sess:
    optimizer = tf.train.GradientDescentOptimizer;
    fit(sess,featureSet,labelSet,loss,optimizer,0.001,2000,10);

    print("acc=",GetAccuracy(sess,featureSet,labelSet,predict_outputs))

    img = cv2.imread("./chepai_test/chepai_test2.jpg");
    # _, img = opencvlib.GetOtusImage(img);
    resize = opencvlib.Resize(img, 60, 20);
    X = np.array([resize.flatten() / 255]);

    prediction = sess.run(predict_outputs, feed_dict={Xholder: X});
    print("prediction", prediction)
    print("W final\n", sess.run(W))
    print("b final\n", sess.run(b))
    savePath = tf.train.Saver().save(sess,"./PlateRecogNet/PlateRecogNet.ckpt")
    print("save path", savePath)







# opencvlib.WaitEscToExit()
# cv2.destroyAllWindows()
