# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x)) #sigmoid的导函数有y' = y(1-y) ,sigmoid函数关于(0,0.5)对称


def ReLU(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x) #tanh的公式为 y=(1-exp(-2x))/(1+exp(-2x)),其导函数为y'=1-y**2,tanh函数关于原点对称
    
input_data = np.random.randn(1000, 100)  # 1000行数据
node_num = 100  # 每层的神经元100个
hidden_layer_size = 5  #5层神经元
activations = {}  # 存每层的输出结果

x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    # 权重初始值实验
    # w = np.random.randn(node_num, node_num) * 1    #标准差为1,sigmoid的输出会趋近与0或1，他的倒数为逐渐接近于0，梯度消失
    w = np.random.randn(node_num, node_num) * 0.01 # 标准差为0.01时，sigmoid的输出趋近于0.5，数据都趋近于一处，表现力受限
    # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num) # Xavier初始值
    # w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num) # He初始值（直观上）可以解释为，因为ReLU的负值区域的值为0，为了使它更有广度，所以需要2倍的系数。


    a = np.dot(x, w)


    # 激活函数
    # z = sigmoid(a)
    z = ReLU(a)
    # z = tanh(a)

    activations[i] = z

# 绘图查看输出的数据分布
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([], [])
    # plt.xlim(0.1, 1)
    # plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()
