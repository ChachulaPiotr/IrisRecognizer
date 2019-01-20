import math
import random
import string
import csv

#random.seed(0)


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

        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)

        # set them to random values
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-1.0, 1.0)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-1.0, 1.0)

    def update(self, inputs):
        if len(inputs) != self.ni - 1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni - 1):
            # self.ai[i] = sigmoid(inputs[i])
            #self.ai[i] = sigmoid(float(inputs[i]))
            self.ai[i] = float(inputs[i])

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + (self.ah[j] * self.wo[j][k])
            #print(str(sum) +"\n")
            self.ao[k] = sigmoid(sum)

        return self.ao[:]

    def outError(self, targets):
        error = []
        for k in range(self.no):
            z = targets[k] - self.ao[k]
            error.append(z)
            #error[k] = targets[k] - self.ao[k]
        return error

    def backPropagate(self, targets, N):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no

        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N * change #+ M * self.co[j][k]
                #self.co[j][k] = change
                # print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N * change #+ M * self.ci[i][j]
                #self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))
            print(self.outError(p[1]))

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=300, N=0.3):
        # N: learning rate
        for i in range(iterations):
            l=0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                error = []
                self.update(inputs)

                fullError = self.backPropagate(targets, N)
                if i % 100 == 0:
                    print('fullError %-.5f' % fullError)
                    error = self.outError(targets)
                    for k in range(len(targets)):
                        print(error[k])
                        print(l)
                        print()
                l = l+1



def demo():
    arr = []

    with open('irisTrainPart.txt') as csv_file:
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
                row[4] = -1.0
            elif(row[4] == 'Iris-versicolor'):
                row[4] = 0.0
            elif(row[4] == 'Iris-virginica'):
                row[4] = 1.0
            else:
                print('Something went wrong')
            arr[-1][1].append(row[4])

    arr2 = []

    with open('irisTestPart2.txt') as csv_file:
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
                row[4] = -1.0
            elif (row[4] == 'Iris-versicolor'):
                row[4] = 0.0
            elif (row[4] == 'Iris-virginica'):
                row[4] = 1.0
            else:
                print('Something went wrong')
            arr2[-1][1].append(row[4])



#    for i in arr:
 #       print(i)

    # create a network with two input, two hidden, and one output nodes
    n = NN(4, 6, 1)
    # train it with some patterns
    n.train(arr)
    # test it
    print()
    print()
    print()
    print()
    n.test(arr2)


if __name__ == '__main__':
    demo()