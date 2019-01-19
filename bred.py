import math
import random
import string
import csv

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


    def update(self, inputs):
        if len(inputs) != self.ni - 1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni - 1):
            # self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = float(inputs[i])

        # hidden activations
        for j in range(self.nh-1):
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
            #print(str(sum) +"\n")
            #self.ao[k] = sigmoid(sum)
            self.ao[k] = sum

        return self.ao[:]

    def backPropagate(self, targets, N):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            #output_deltas[k] = dsigmoid(self.ao[k]) * error
            output_deltas[k] = error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                #error = error + output_deltas[k] * self.wo[j][k]
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
        for p in patterns:
            print(p[0], '->', self.update(p[0]), '==', round(self.update(p[0])[0]), '->', p[1])

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
        # M: momentum factor
        for i in range(iterations):
            #error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                self.backPropagate(targets, N)
                # print('error %-.5f' % self.backPropagate(targets, N))
            #if i % 100 == 0:
                #print('error %-.5f' % error)


def demo():
    arr = []
    arr2 = []

    with open('iris2.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            #print(row[0] + " " + row[1])
            #print(row[2] + " " + row[3])
            #print(row[4] + "\n")
            arr.append([])      # array for one flower
            arr[-1].append([])  # array for input data
            arr[-1].append([])  # array for output data

            arr[-1][0].append(row[0])
            arr[-1][0].append(row[1])
            arr[-1][0].append(row[2])
            arr[-1][0].append(row[3])
            if(row[4] == 'Iris-setosa'):
                row[4] = 1.0
            elif(row[4] == 'Iris-versicolor'):
                row[4] = 2.0
            elif(row[4] == 'Iris-virginica'):
                row[4] = 3.0
            else:
                print('Something went wrong')
            arr[-1][1].append(row[4])

    with open('iris.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            #print(row[0] + " " + row[1])
            #print(row[2] + " " + row[3])
            #print(row[4] + "\n")
            arr2.append([])      # array for one flower
            arr2[-1].append([])  # array for input data
            arr2[-1].append([])  # array for output data

            arr2[-1][0].append(row[0])
            arr2[-1][0].append(row[1])
            arr2[-1][0].append(row[2])
            arr2[-1][0].append(row[3])
            if(row[4] == 'Iris-setosa'):
                row[4] = 1.0
            elif(row[4] == 'Iris-versicolor'):
                row[4] = 2.0
            elif(row[4] == 'Iris-virginica'):
                row[4] = 3.0
            else:
                print('Something went wrong')
            arr2[-1][1].append(row[4])


    # for i in arr:
    #     print(i)

    # create a network with two input, two hidden, and one output nodes
    n = NN(4, 5, 1)
    # train it with some patterns
    #n.test(arr2)
    n.train(arr2,400,0.2)
    # test it
    #n.test(arr2)
    #p = ([6.2, 3.4, 5.3, 2.3], [3.0])
    p = ([5.9, 3.0, 5.1, 1.8], [3.0])
    print(p[0], '->', n.update(p[0]), '==', round(n.update(p[0])[0]), '->', p[1])

if __name__ == '__main__':
    demo()