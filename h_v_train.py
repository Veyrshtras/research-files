import sys
import traceback

from scipy.special import expit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Kerakli funksiyalar yani sigmoid fun-ya
def sigmoid(x):
    return expit(x)


# sigmoid fun-yaning hosilasi olindi
def derivative_sigmoid(x):
    return np.array(x) * (1 - np.array(x))


#  neuro network class

class Layer:
    def __init__(self, n_inputs, m_neurons):
        self.wights = np.random.randn(n_inputs, m_neurons)
        self.biases = np.zeros((1, m_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = sigmoid(np.dot(inputs, self.wights) + self.biases)

    def backpropagation(self, output):
        A = 2 * (np.array(output) - np.array(self.output)) * np.array(derivative_sigmoid(self.output))
        B = [np.array(self.inputs).ravel()]
        d_weights = np.dot(np.array(B).T, np.array(A))
        d_biases = A
        d_inputs = np.dot(np.array(A), np.array(self.wights).T)
        self.inputs = np.array(self.inputs) + eta * np.array(d_inputs)
        self.biases = np.array(self.biases) + eta * np.array(d_biases)
        self.wights = np.array(self.wights) + eta * np.array(d_weights)


## Network

layer1 = Layer(2, 10)
layer2 = Layer(10, 10)
layer3 = Layer(10, 1)

## Training
predict = 0
E = 0.0005
eta = 1

h='C:/Users/User/Desktop/proekt files for univer/h.xlsx'
v='C:/Users/User/Desktop/proekt files for univer/v.xlsx'

arrH = np.array(pd.read_excel(h))
arrV = np.array(pd.read_excel(v))
print(arrH)
print("\n\n\nafter arrH rinted\n\n\n")
print(arrV)
t = 60 / 500
trainX = [[0.0 for i in range(2)] for item in range(485)]
trainY = [[0.0 for i in range(1)] for item in range(485)]

for item in range(485):
    trainX[item][0] = item * t
    trainX[item][1] = arrV[item][0]
    trainY[item][0] = arrH[item][0]

testX = [[0.0 for i in range(2)] for item in range(15)]
testY = [[0.0 for i in range(1)] for item in range(15)]

for item in range(15):
    testX[item][0] = (item + 485) * t
    testX[item][1] = arrV[item + 485][0]
    testY[item][0] = arrH[item + 485][0]
a = np.zeros((2,500))
for i in range(485):
    InputX = np.array(trainX[i])
    OutputY = np.array(trainY[i])

    layer1.forward(InputX)
    layer2.forward(layer1.output)
    layer3.forward(layer2.output)
    a[0][i] = layer3.output[0][0]
    a[1][i] = trainY[i][0]
    print(layer3.output[0][0], "->layer", i, "/n", trainY[i][0], "->testY", i)
    if abs(layer3.output[0][0] - trainY[i][0]) < E:
        predict = predict + 1
    layer3.backpropagation(OutputY)
    layer2.backpropagation(layer3.inputs)
    layer1.backpropagation(layer2.inputs)

alpha = predict
print("after train result 1: ", alpha / 485)

##Test
predict = 0
for item in range(15):
    InputX = np.array(testX[item])
    layer1.forward(InputX)
    layer2.forward(layer1.output)
    layer3.forward(layer2.output)
    a[0][item+485] = layer3.output[0][0]
    a[1][item+485] = testY[item][0]
    print(layer3.output[0][0],"->layer",item,"/n",testY[item][0],"->testY",item)
    if abs(layer3.output[0][0] - testY[item][0]) < E:
        predict = predict + 1
        
        
print("after train result 2: ", predict / 15)
def chizma(a):
    try:


        hh = a
        fig_balandlik = 1.5

        fig, ax = plt.subplots(figsize=(50, 4))

        y = list(hh[0])

        # print(y)

        ax.plot(range(len(y)), y, label="H*" )
        y = list(hh[1])
        ax.plot(range(len(y)), y, label="H" )
        ax.set_title("высота")

        # ax.plot(xs, bs(xs), label="b")
        # ax.plot(xs, us(xs), label="univar")

        # ax.set_xlim(-10,10)
        ax.legend(loc='lower left', ncol=2)
        plt.show()
    except BaseException as ex:
        # Get current system exception
        ex_type, ex_value, ex_traceback = sys.exc_info()

        # Extract unformatter stack traces as tuples
        trace_back = traceback.extract_tb(ex_traceback)

        # Format stacktrace
        stack_trace = list()

        for trace in trace_back:
            stack_trace.append(
                "File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))

        print("Exception type : %s " % ex_type.__name__)
        print("Exception message : %s" % ex_value)
        print("Stack trace : %s" % stack_trace)
alpha = predict
chizma(a)