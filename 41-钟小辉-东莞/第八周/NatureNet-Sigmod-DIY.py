import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import pandas as pd


class NatureNetWork:
    # 构建网络节点
    def __init__(self, inputNode, hiddenNode, outputNode, learningRate):
        self._inputNode = inputNode
        self._hiddenNode = hiddenNode
        self._outputNode = outputNode

        """
             初始化权重矩阵，我们有两个权重矩阵，一个是wih表示输入层和中间层节点间链路权重形成的矩阵
             一个是who, 表示中间层和输出层间链路权重形成的矩阵
        """
        self._wih = np.random.rand(self._hiddenNode, self._inputNode) - 0.5
        self._bih = np.zeros((self._hiddenNode, 1))
        self._who = np.random.rand(self._outputNode, self._hiddenNode) - 0.5
        self._bho = np.zeros((self._outputNode, 1))

        # 设置学习率
        self._learningRate = learningRate


        '''
              scipy.special.expit对应的是sigmod函数.
              lambda是Python关键字，类似C语言中的宏定义，当我们调用self.activation_function(x)时，编译器会把其转换为spicy.special_expit(x)。
        '''
        # 设置激活函数,为sigmod函数,也可以手写
        self._activate_function2 = lambda a: scipy.special.expit(a)

        # 手写,比对计算结果看，应该是一样的
        self._activate_function = lambda x: self.activate_function(x)

        # 这里手写一遍sigmod函数

    def activate_function(self, x, deriv=False):
        if (deriv):  # 导数
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def train(self, x, y):
        # 根据输入的训练数据更新节点链路权重
        '''
        把inputs_list, targets_list转换成numpy支持的二维矩阵
        .T表示做矩阵的转置
        '''
        inputs = np.array(x, ndmin=2).reshape((-1,1))
        label = np.array(y, ndmin=2).reshape((-1,1))
        # print(inputs.shape)
        # print(label.shape)
        hidden_Input = np.dot(self._wih, inputs) + self._bih
        hidden_Output = self._activate_function(hidden_Input)
        final_Input = np.dot(self._who, hidden_Output) + self._bho
        final_outputs = self._activate_function(final_Input)

        # 计算损失函数
        loss = label - final_outputs
        correct_loss = label[np.argmax(label)] - final_outputs[np.argmax(label)]

        #注意，这里返回计算dhidden_error的loss必须与hidden的维数一致，所以是 _who.T
        dhidden_error = np.dot(self._who.T, (label - final_outputs) * self.activate_function(final_outputs, True))

        #同理
        dw2 =  np.dot((label - final_outputs) * self.activate_function(final_outputs,True) , np.transpose(hidden_Output))
        db2 =  (label - final_outputs) * self.activate_function(final_outputs,True)


        dw1 = np.dot(dhidden_error , np.transpose(inputs))
        db1 = dhidden_error


        # dinputs =  np.dot(dhidden_error , self._wih.T)

        self._wih += self._learningRate * dw1
        self._bih += self._learningRate * db1
        self._who += self._learningRate * dw2
        self._bho += self._learningRate * db2

        return correct_loss
        # print(final_outputs)
        # 求softmax

    def predict(self, input):
        hidden_Input = np.dot(self._wih, input) + self._bih
        hidden_Output = self._activate_function(hidden_Input)

        final_Input = np.dot(self._who, hidden_Output) + self._bho
        final_outputs = self._activate_function(final_Input)

        # print(self._activate_function2(final_Input))
        # print(final_outputs)
        return final_outputs


if __name__ == "__main__":
    input_Node = 784
    hidden_Node = 200
    output_Node = 10
    learning_Rate = 0.05
    reg = 1e-3  # regularization strength

    # 建立神经网络
    net = NatureNetWork(input_Node, hidden_Node, output_Node, learning_Rate)

    # 读取数据
    data_file = open("dataset/mnist_train.csv")
    train_list = data_file.readlines()
    data_file.close()
    # print(len(train_list))
    # print(train_list)

    # 分成10类
    # _train_image = np.zeros((input_Node * 10 ,len(train_list)))
    # _train_label = np.zeros((1,len(train_list)))
    _train_image =[]
    _train_label =[]
    for i in range(len(train_list)):
        _trainValue = train_list[i].split(',')
        _image1 = np.asfarray(_trainValue[1:]).reshape((28,28))
        _train_image.append(_image1)
        _temp1 = np.zeros((output_Node,1))
        _temp1[int(_trainValue[0])] = 0.99
        _train_label.append(_temp1)

    data_file2 = open("dataset/mnist_test.csv")
    test_list = data_file2.readlines()
    data_file2.close()
    _test_image = []
    _test_label = []
    # print(len(test_list))
    for i in range(len(test_list)):
        _testValue = test_list[i].split(',')
        _image2 = np.asfarray(_testValue[1:]).reshape((-1, 1))
        _test_image.append(_image2)

        _temp2 = np.zeros((output_Node, 1))
        _temp2[int(_testValue[0])] = 0.99
        _test_label.append(_temp2)


    # #开始训练
    eporch = 100
    for k in range(eporch):
        for i in range(len(_train_image)):
            loss = net.train(_train_image[i], _train_label[i])
            if i % (len(_train_image))== 0:
                print("第{}百次的损失值为{}",k,loss)

    arrc = []

    for i in range(len(_test_image)):
        output = net.predict(_test_image[i])
        # image_array = _test_image[i].reshape((28, 28))
        # plt.imshow(image_array, cmap='Greys', interpolation='None')
        # plt.show()
        print("预测该数值为{}",np.argmax(output))
        print("实际该数值为{}",np.argmax(_test_label[i]))
        if np.argmax(output) ==np.argmax(_test_label[i]):
            arrc.append(1)
        else:
            arrc.append(0)
    print(arrc)

    # 计算图片判断的成功率
    scores_array = np.asarray(arrc)
    print("perfermance = ", scores_array.sum() / scores_array.size)
