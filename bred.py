import math
import random
import string
import csv
from tabulate import tabulate
import numpy as np

random.seed(0)


# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b - a) * random.random() + a


# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill] * J)
    return m


# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)


# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - math.tanh(y) ** 2


class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1  # +1 for bias node
        self.nh = nh + 1  # +1 for bias node
        self.no = no

        # activations for nodes
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no
        self.sh = [1.0] * self.nh
        self.so = [1.0] * self.no

        # normalization parameters
        self.max = [0.0] * (self.ni - 1)
        self.min = [10.0] * (self.ni - 1)

        # create weights
        self.wi = makeMatrix(self.nh, self.ni)
        self.wo = makeMatrix(self.no, self.nh)
        # set them to random values
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[j][i] = rand(-1.0, 1.0)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[k][j] = rand(-1.0, 1.0)

        # self.wi[0][0] = 11.5349
        # self.wi[1][0] = 96.8871
        # self.wi[2][0] = -0.6602
        # self.wi[3][0] = 58.5381
        # self.wi[4][0] = -0.5916
        # self.wi[0][1] = 2.3221
        # self.wi[1][1] = -52.9829
        # self.wi[2][1] = 3.9816
        # self.wi[3][1] = -37.0987
        # self.wi[4][1] = -47.8942
        # self.wi[0][2] = 13.7675
        # self.wi[1][2] = 37.0570
        # self.wi[2][2] = -5.6558
        # self.wi[3][2] = 47.6512
        # self.wi[4][2] = 77.1399
        # self.wi[0][3] = 3.5308
        # self.wi[1][3] = -40.7752
        # self.wi[2][3] = -10.4035
        # self.wi[3][3] = -39.7127
        # self.wi[4][3] = 158.3520
        # self.wi[0][4] = -13.3867
        # self.wi[1][4] = -42.9766
        # self.wi[2][4] = -6.8914
        # self.wi[3][4] = -31.6713
        # self.wi[4][4] = -92.5038
        # self.wo[0][0] = 0.0001
        # self.wo[0][1] = -3.1652
        # self.wo[0][2] = -0.5
        # self.wo[0][3] = 3.1652
        # self.wo[0][4] = 0.5
        # self.wo[0][5] = 2
        # self.max[0] = 7.9
        # self.max[1] = 4.4
        # self.max[2] = 6.9
        # self.max[3] = 2.5
        # self.min[0] = 4.3
        # self.min[1] = 2.0
        # self.min[2] = 1.0
        # self.min[3] = 0.1

    def update(self, inputs):
        inputs = self.normalize(inputs)
        if len(inputs) != self.ni - 1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni - 1):
            # self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh - 1):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[j][i]
            self.ah[j] = sigmoid(sum)
            self.sh[j] = sum

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + (self.ah[j] * self.wo[k][j])
            # print(str(sum) +"\n")
            #self.ao[k] = sigmoid(sum)
            #self.so[k] = self.ao[k]
            self.ao[k] = sum
        if self.ao[0] > 1000:
            print(self.ao[0])
        return self.ao[:]

    def backPropagate(self, targets, N):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            #output_deltas[k] = dsigmoid(self.so[k]) * error
            output_deltas[k] = error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                # error = error + output_deltas[k] * self.wo[j][k]
                error = error + output_deltas[k] * self.wo[k][j]
            hidden_deltas[j] = dsigmoid(self.sh[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[k][j] = self.wo[k][j] + N * change
                # print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[j][i] = self.wi[j][i] + N * change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
           error = error + 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def test(self, patterns):
        mb = makeMatrix(3, 3, 0.0)
        for p in patterns:
            print(p[0], '->', self.update(p[0]), '==', round(self.update(p[0])[0]), '->', p[1])
            if (p[1][0] == 1) & (round(self.ao[0]) == 1):
                mb[0][0] = mb[0][0] + 1
            elif (p[1][0] == 1) & (round(self.ao[0]) == 2):
                mb[1][0] = mb[1][0] + 1
            elif (p[1][0] == 1) & (round(self.ao[0]) == 3):
                mb[2][0] = mb[2][0] + 1
            elif (p[1][0] == 2) & (round(self.ao[0]) == 1):
                mb[0][1] = mb[0][1] + 1
            elif (p[1][0] == 2) & (round(self.ao[0]) == 2):
                mb[1][1] = mb[1][1] + 1
            elif (p[1][0] == 2) & (round(self.ao[0]) == 3):
                mb[2][1] = mb[2][1] + 1
            elif (p[1][0] == 3) & (round(self.ao[0]) == 1):
                mb[0][2] = mb[0][2] + 1
            elif (p[1][0] == 3) & (round(self.ao[0]) == 2):
                mb[1][2] = mb[1][2] + 1
            elif (p[1][0] == 3) & (round(self.ao[0]) == 3):
                mb[2][2] = mb[2][2] + 1
                print('sdfa')
        self.printmb(mb)

    def printmb(self, mb):
        print(tabulate([['', 'Setosa', mb[0][0], mb[0][1], mb[0][2]],
                        ['Predicted', 'Versicolor', mb[1][0], mb[1][1], mb[1][2]],
                        ['Class', 'Virginica', mb[2][0], mb[2][1], mb[2][2]]],
                       headers=['\nSetosa', 'Actual Class\nVersicolor', '\nVirginica'],
                       tablefmt="plain"))

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=400, N=0.1):
        # N: learning rate
        for p in patterns:
            for i in range(self.ni - 1):
                if float(p[0][i]) > self.max[i]:
                    self.max[i] = float(p[0][i])
                if float(p[0][i]) < self.min[i]:
                    self.min[i] = float(p[0][i])
        for i in range(iterations):
            error = 0.0
            k = 0
            for p in patterns:
                k = k+1
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                #error = self.backPropagate(targets, N)
                error = error + self.backPropagate(targets, N)
                #error = error + 0.5 * (targets[0] - self.ao[0]) ** 2
                #print(error, '->', targets[0], '->', self.ao[0])
                #print('error %-.5f' % self.backPropagate(targets, N))
                #if (k == 10):
                    #self.backPropagate(targets, N, error / len(patterns))
            #self.backPropagate(targets, N, error/len(patterns))
            if i % 10 == 0:
                print('error %-.5f' % error)

    def normalize(self, inputs, ru=1, rd=-1):
        ninputs = [0.0] * (self.ni - 1)
        for i in range(self.ni - 1):
            ninputs[i] = ((ru - rd) * (float(inputs[i]) - self.min[i]) / (self.max[i] - self.min[i])) + rd
        return ninputs


def demo():
    arr = []
    arr2 = []

    with open('iris2.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            # print(row[0] + " " + row[1])
            # print(row[2] + " " + row[3])
            # print(row[4] + "\n")
            arr.append([])  # array for one flower
            arr[-1].append([])  # array for input data
            arr[-1].append([])  # array for output data

            arr[-1][0].append(row[0])
            arr[-1][0].append(row[1])
            arr[-1][0].append(row[2])
            arr[-1][0].append(row[3])
            if (row[4] == 'Iris-setosa'):
                row[4] = 1.0
            elif (row[4] == 'Iris-versicolor'):
                row[4] = 2.0
            elif (row[4] == 'Iris-virginica'):
                row[4] = 3.0
            else:
                print('Something went wrong')
            arr[-1][1].append(row[4])

    with open('iris.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            # print(row[0] + " " + row[1])
            # print(row[2] + " " + row[3])
            # print(row[4] + "\n")
            arr2.append([])  # array for one flower
            arr2[-1].append([])  # array for input data
            arr2[-1].append([])  # array for output data

            arr2[-1][0].append(row[0])
            arr2[-1][0].append(row[1])
            arr2[-1][0].append(row[2])
            arr2[-1][0].append(row[3])
            if (row[4] == 'Iris-setosa'):
                row[4] = 1.0
            elif (row[4] == 'Iris-versicolor'):
                row[4] = 2.0
            elif (row[4] == 'Iris-virginica'):
                row[4] = 3.0
            else:
                print('Something went wrong')
            arr2[-1][1].append(row[4])
            # setosa = makeMatrix(50, 4, 0.0)
            # versicolor = makeMatrix(50, 4, 0.0)
            # virginica = makeMatrix(50, 4, 0.0)
            setosa = []
            versicolor = []
            virginica = []
            k = 0
            for r in arr2:
                k = k + 1
                if k <= 50:
                    setosa.append(r)
                elif k <= 100:
                    versicolor.append(r)
                else:
                    virginica.append(r)
        print(setosa)

    # for i in arr:
    #     print(i)

    # create a network with four input, five hidden, and one output nodes
    n = NN(4, 5, 1)
    # train it with some patterns
    #n.test(arr2)

    #setting ratio between teaching and veryfing collection
    border = 95
    #
    border = round(50*border/100)
    teach = []
    verify = []
    rows = np.random.permutation(50)
    for i in range(border):
        teach.append(setosa[rows[i]])
    for i in range(border, 50):
        verify.append(setosa[rows[i]])
    for i in range(border):
        teach.append(versicolor[rows[i]])
    for i in range(border, 50):
        verify.append(versicolor[rows[i]])
    for i in range(border):
        teach.append(virginica[rows[i]])
    for i in range(border, 50):
        verify.append(virginica[rows[i]])
    n.train(teach, 1000, 0.05)
    # test it
    n.test(verify)
    # p = ([6.2, 3.4, 5.3, 2.3], [3.0])
    #p = ([5.9, 3.0, 5.1, 1.8], [3.0])
    #l = ([5.2, 2.7, 2.7, 4.0], [2.0])
    #z = ([5.7, 4.4, 4.4, 1.5], [1.0])
    #m = ([4.6, 3.2, 3.2, 1.4], [1.0])
    #print(p[0], '->', n.update(p[0]), '==', round(n.update(p[0])[0]), '->', p[1])
    #p = ([6.0, 2.7, 5.1, 1.6], [2.0])
    #print(p[0], '->', n.update(p[0]), '==', round(n.update(p[0])[0]), '->', p[1])
    #print(l[0], '->', n.update(l[0]), '==', round(n.update(l[0])[0]), '->', l[1])
    #print(z[0], '->', n.update(z[0]), '==', round(n.update(z[0])[0]), '->', z[1])
    #print(m[0], '->', n.update(m[0]), '==', round(n.update(m[0])[0]), '->', m[1])
    #print(n.wi)


if __name__ == '__main__':
    demo()
