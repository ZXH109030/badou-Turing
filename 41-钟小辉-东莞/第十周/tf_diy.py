import warnings
warnings.filterwarnings("ignore")
import pickle
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder


#1.获取图片数据及label
def get_cifar10_data(file):
    with open(file,'rb') as f:
        data = pickle.load(f,encoding='ISO-8859-1')
    return data

train_labels = [] #存放训练图片分类
test_labels = []
x_train = [] #存放训练图片
for i in range(1,6):
    data = get_cifar10_data("D://cifar-10-python//cifar-10-batches-py//data_batch_%d"%i)
    train_labels.append(data["labels"])
    x_train.append(data["data"])

#读取的数据有 规定的格式
x_train = np.array(x_train)
x_train = np.transpose(x_train.reshape(-1,3,32,32),[0,2,3,1]).reshape(-1,3072)
y_train = np.array(train_labels)
y_train = y_train.reshape(-1)

#根据实际使用情况看，是否转换Onehote
one_hot = OneHotEncoder()
y_train_dim = one_hot.fit_transform(y_train.reshape(-1,1)).toarray()

#测试数据
data_test = get_cifar10_data("D://cifar-10-python//cifar-10-batches-py//test_batch")
test_labels = data_test["labels"]
x_test = data_test["data"]
x_test = np.transpose(np.array(x_test).reshape(-1,3,32,32),[0,2,3,1]).reshape(-1,3072)
y_test = np.array(test_labels).reshape(-1)
y_test_dim = one_hot.fit_transform(y_test.reshape(-1,1)).toarray()

# digit = x_train[2].reshape(32,32,3)
# import matplotlib.pyplot as plt
# plt.imshow(digit,cmap = plt.cm.binary)
# plt.show()

# 2.定义占位符
x = tf.placeholder(dtype= tf.float32,shape =[None,3072])
y = tf.placeholder(dtype= tf.float32,shape =[None,10] )
kp = tf.placeholder(dtype=tf.float32)

# 3.构建conv网络
def net_work(X, kp):
    _input  = tf.reshape(X,[-1,32,32,3])

    #第一层
    filter1 = gen_kernel(shape=[3,3,3,64])
    bias1 = gen_kernel(shape= [64])
    # bias1=tf.Variable(tf.constant(0.0,shape=[64]))
    pool1 = conv_relu_pool(_input, bias1, filter1)

    #第二层
    filter2 = gen_kernel(shape=[3,3,64,128])
    bias2 = gen_kernel(shape= [128])
    # bias2=tf.Variable(tf.constant(0.0,shape=[128]))
    pool2 = conv_relu_pool(pool1, bias2, filter2)

    #第二层
    filter3 = gen_kernel(shape=[3,3,128,256])
    bias3 = gen_kernel(shape= [256])
    # bias2=tf.Variable(tf.constant(0.0,shape=[128]))
    pool3 = conv_relu_pool(pool2, bias3, filter3)

    #全连接层
    #拍扁矩阵
    dense = tf.reshape(pool3,shape=[-1,4 * 4 * 256])
    w1 = gen_kernel(shape=[4*4*256,1024])
    b1 = gen_kernel(shape=[1024])
    fc1 = tf.layers.batch_normalization(tf.matmul(dense,w1) + b1,training= True)
    # fc1 = tf.matmul(dense, w1) + b1
    fc_1 = tf.nn.relu(fc1)

    dp = tf.nn.dropout(fc_1,keep_prob=kp)

    # dense2 = tf.reshape(dp,shape=[-1,1024])
    # w2 = gen_kernel(shape=[1024,256])
    # b2 = gen_kernel(shape=[256])
    # fc2 = tf.layers.batch_normalization(tf.matmul(dp,w2) + b2,training= True)
    # # fc2 = tf.matmul(dp1, w2) + b2
    # fc_2 = tf.nn.relu(fc2)
    # dp2 = tf.nn.dropout(fc_2, keep_prob=kp)

    #输出层
    out_w = gen_kernel(shape=[1024,10])
    out_b = gen_kernel(shape=[10])
    out = tf.matmul(dp,out_w)+out_b
    return out


def conv_relu_pool(_input, bias, filter):
    conv = tf.nn.conv2d(_input, filter, [1, 1, 1, 1], padding="SAME") + bias
    conv = tf.layers.batch_normalization(conv,training=True)
    relu = tf.nn.relu(conv)
    pool = tf.nn.max_pool(relu, [1, 3, 3, 1], [1, 2, 2, 1], "SAME")
    return  pool


def gen_kernel(shape,stddev = 5e-2):
    kernel = tf.Variable(tf.truncated_normal(shape, stddev))
    return kernel


out = net_work(x,kp)

#交叉熵
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y,1),logits=out))
#准确率
_y = tf.nn.softmax(out)
equal = tf.equal(tf.argmax(y,axis=-1),tf.argmax(_y,axis=1))
accuracy = tf.reduce_mean(tf.cast(equal,tf.float32))
opt = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

saver = tf.train.Saver()
# 从总数据中获取一批数据
index = 0
def next_batch(X, y):
    global index
    batch_X = X[index*128:(index + 1)*128]
    batch_y = y[index*128:(index + 1)*128]
    index += 1
    if X.shape[0] > 49999:
        if index == 390:
            index = 0
    else:
        if index == 50:
            index = 0
    return batch_X, batch_y

EPOCHES = 100
EPOCHES_TEST = 50
with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    saver.restore(sess, './model/estimator-91')
    for i in range(EPOCHES):
        batch_X, batch_y = next_batch(x_train, y_train_dim)
        opt_, loss_ ,score_train = sess.run([opt, loss, accuracy], feed_dict={x: batch_X, y: batch_y, kp: 0.5})
        print('iter count:%d, mini_batch loss:%0.4f,train accuracy:%0.4f' % (i + 1, loss_, score_train))

        if score_train > 0.7:  # 当训练准确率达到0.6时保存模型
            saver.save(sess, './model/estimator', i + 1)
            # break
    saver.save(sess, './model/estimator', i + 1)  # 如果都小于0.6，则保存最后一个模型
    index = 0
    scores = 0
    for i in range(EPOCHES_TEST):
        batch_X_test, batch_y_test = next_batch(x_test, y_test_dim)
        score_test = sess.run(accuracy, feed_dict={x: batch_X_test, y: batch_y_test, kp: 1.0})  # 判断测试数据的acc
        scores += np.sum(score_test)
        print('test accuracy:%0.4f', scores/(i+1))
    # score_test = sess.run(accuracy, feed_dict={X: X_test, y: y_test, kp: 1.0})  # 判断测试数据的acc
