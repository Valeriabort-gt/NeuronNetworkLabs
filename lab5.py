import math
import numpy as np
import matplotlib.pyplot as plt
from random import random
from tkinter import *
from tkinter import messagebox
firstTrain = False
secondTrain = False


def main():
    mainWindow = Tk()
    mainWindow.resizable(width=False, height=False)
    mainWindow.title("Лабораторная работа 5")
    screenWidth = mainWindow.winfo_screenwidth()
    screenHeight = mainWindow.winfo_screenheight()
    x = (screenWidth - 600) / 2
    y = (screenHeight - 500) / 2
    mainWindow.geometry('%dx%d+%d+%d' % (600, 500, x, y))
    mainWindow.configure(background="#fff")
    dataset = load_txt("train.txt")
    testset = load_txt("test.txt")
    mean = np.mean(dataset)
    std = np.std(dataset)
    sequence_len = 100
    # Создание меню
    xCoordinate = 600/2-230/2
    buttonColor = "#91e2e3"
    buttonWidth = "25"
    button1 = Button(text="Задание 1", command=lambda: first_Task(), bg=buttonColor, fg="white",
                     width=buttonWidth, height="1", font=("MS Sans Serif", 14), relief="groove", activebackground="#dc3f3f")
    button1.place(x=xCoordinate, y=90)
    button2 = Button(text="Задание 2", command=lambda: second_Task(sequence_len, dataset, testset), bg=buttonColor,
                     fg="white", width=buttonWidth, height="1", font=("MS Sans Serif", 14), relief="groove", activebackground="#dc3f3f")
    button2.place(x=xCoordinate, y=140)
    button3 = Button(text="Обучить сети задания 1", command=lambda: train_first_Networks(), bg=buttonColor, fg="white",
                     width=buttonWidth, height="1", font=("MS Sans Serif", 14), relief="groove", activebackground="#dc3f3f")
    button3.place(x=xCoordinate, y=190)
    button4 = Button(text="Обучить сети задания 2", command=lambda: train_second_Networks(sequence_len, dataset), bg=buttonColor,
                     fg="white", width=buttonWidth, height="1", font=("MS Sans Serif", 14), relief="groove", activebackground="#dc3f3f")
    button4.place(x=xCoordinate, y=240)
    mainWindow.mainloop()


def function_x(x_list):
    result = []
    for x in x_list:
        result.append(np.exp(np.cos(x)*np.sin(x)))
    return result


class RBF_NeuralNetwork:
    def __init__(self, hidden_shape, sigma=1.0):
        self.hidden_shape = hidden_shape
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def rbf_function(self, center, data_point):
        return np.exp(np.linalg.norm((data_point - center) ** 2 / (2 * self.sigma ** 2)))

    def calculate_interpolation_matrix(self, x):
        matrix = np.zeros((len(x), self.hidden_shape))
        for i, point in enumerate(x):
            for j, center in enumerate(self.centers):
                matrix[i, j] = self.rbf_function(point, center)
        return matrix

    def select_centers(self, y):
        random_args = np.random.choice(len(y), self.hidden_shape)
        centers = y[random_args]
        return centers

    def fit(self, x, y):
        self.centers = self.select_centers(x)
        matrix = self.calculate_interpolation_matrix(x)
        self.weights = np.dot(np.linalg.pinv(matrix), y)

    def predict(self, x):
        matrix = self.calculate_interpolation_matrix(x)
        predictions = np.dot(matrix, self.weights)
        return predictions


class Perceptron:
    def __init__(self, l_rate=0.1):
        self.layers = []
        self.l_rate = l_rate
        self.out = 0

    def append_layer(self, n_inputs, n_output, existBias=False, isActivate=False):
        self.layers.append(PerceptronLayer(
            n_inputs, n_output, existBias, isActivate))

    def forward_propagate(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward_propagate(out)
        self.out = out
        return out

    def backward_propagate(self, y):
        error = self.out - y
        for i in reversed(range(len(self.layers))):
            if i != len(self.layers)-1:
                self.layers[i].backward_propagate(
                    self.layers[i+1].neurons, False, error)
            else:
                self.layers[i].backward_propagate(None, True, y)

    def update_weights(self, row):
        for i in range(len(self.layers)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output']
                          for neuron in self.layers[i - 1].neurons]
                if i != 1:
                    oldInputs = inputs
                    inputs = []
                    for j in range(len(oldInputs)):
                        current = oldInputs[j]
                        previousInputs = [neuron['output']
                                          for neuron in self.layers[i - 2].neurons]
                        sumIn = 0
                        for item in previousInputs:
                            sumIn += item * current
                        inputs.append(sumIn)
            for neuron in self.layers[i].neurons:
                for j in range(len(inputs)):
                    neuron['weights'][j] -= self.l_rate * \
                        neuron['delta'] * inputs[j]
                neuron['weights'][-1] -= self.l_rate * neuron['delta']

    def fit(self, dataset):
        for row in dataset:
            self.forward_propagate(row)
            self.backward_propagate(row[-1])
            self.update_weights(row)

    def predict(self, x_list):
        result = []
        for x in x_list:
            result.append(self.forward_propagate([x, None]))
        return result


class PerceptronLayer:
    def __init__(self, n_inputs, n_output, existBias=False, isActivate=False):
        if existBias:
            n_inputs += 1
        self.neurons = [{'weights': [np.random.uniform(-np.sqrt(1 / n_inputs), np.sqrt(1 / n_inputs)) for i in range(n_inputs)]}
                        for i in range(n_output)]
        self.existBias = existBias
        self.isActivate = isActivate

    def transfer(self, activation):
        if self.isActivate:
            return 1.0 / (1.0 + math.exp(-activation))
        return activation

    def transfer_derivative(self, output):
        if self.isActivate:
            return output * (1.0 - output)
        return 1

    def activate(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights)-1):
            activation += weights[i] * inputs[i]
        return activation

    def forward_propagate(self, x):
        new_inputs = []
        for neuron in self.neurons:
            activation = self.activate(neuron['weights'], x)
            neuron['output'] = self.transfer(activation)
            new_inputs.append(neuron['output'])
        return new_inputs

    def backward_propagate(self, nextNeurons=None, isLast=False, y=None):
        errors = list()
        if not isLast:
            for j in range(len(self.neurons)):
                error = 0.0
                for neuron in nextNeurons:
                    error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
        else:
            for j in range(len(self.neurons)):
                neuron = self.neurons[j]
                errors.append(neuron['output'] - y)
        for j in range(len(self.neurons)):
            neuron = self.neurons[j]
            neuron['delta'] = errors[j] * \
                self.transfer_derivative(neuron['output'])


class RecurrentNeuralNetwork:
    def __init__(self, input_number, hidden_number, output_number, l_rate=0.1):
        self.l_rate = l_rate
        self.w1 = np.random.uniform(-np.sqrt(1 / input_number), np.sqrt(
            1 / input_number), size=[input_number, hidden_number])
        self.b1 = np.random.uniform(size=[1, hidden_number])
        self.w2 = np.random.uniform(-np.sqrt(1 / hidden_number), np.sqrt(
            1 / hidden_number), size=[hidden_number, hidden_number])
        self.b2 = np.random.uniform(size=[1, hidden_number])
        self.h = np.zeros(shape=[1, hidden_number])
        self.h_prev = np.zeros(shape=[1, hidden_number])
        self.w3 = np.random.uniform(-np.sqrt(1 / hidden_number), np.sqrt(
            1 / hidden_number), size=[hidden_number, output_number])

    def transfer(self, x):
        return np.tanh(x)

    def transfer_derivative(self, output):
        return (1 - np.tanh(output) ** 2)

    def forward_propagate(self, x):
        self.x = x
        self.h_prev = self.h
        self.h = np.dot(self.x, self.w1) + self.b1 + \
            np.dot(self.h_prev, self.w2) + self.b2
        self.h = self.transfer(self.h)
        self.output = np.dot(self.h, self.w3)
        return self.output

    def backward_propagate(self, y):
        error = self.output - y
        self.dw = np.dot(self.h.T, error)
        delta = self.transfer_derivative(self.h) * np.dot(error, self.w3.T)
        self.dw_w1 = np.dot(self.x.T, delta)
        self.db_b1 = delta
        self.dw_w2 = np.dot(self.h_prev.T, delta)
        self.db_b2 = delta

    def update_weights(self):
        self.w1 -= self.l_rate * self.dw_w1
        self.b1 -= self.l_rate * self.db_b1
        self.w2 -= self.l_rate * self.dw_w2
        self.b2 -= self.l_rate * self.db_b2
        self.w3 -= self.l_rate * self.dw


class SlidingLayer:
    def __init__(self, n_inputs, n_output, isActivate=False):

        self.weight = np.random.uniform(low=-np.sqrt(1 / n_inputs),
                                        high=np.sqrt(1 / n_inputs), size=[n_inputs, n_output])

        self.isActivate = isActivate

    def transfer(self, activation):
        if self.isActivate:
            return np.tanh(activation)
        return activation

    def transfer_derivative(self, output):
        if self.isActivate:
            return 1 - np.tanh(output) ** 2
        return 1

    def forward_propagate(self, x):
        self.input = x
        self.not_activated = np.dot(x, self.weight)
        self.out = self.transfer(self.not_activated)
        return self.out

    def backward_propagate(self, error):
        delta = error * self.transfer_derivative(self.not_activated)
        self.dw = np.dot(self.input.T, delta)
        return np.dot(delta, self.weight.T)


class SlidingPerceptron:
    def __init__(self, l_rate=0.1):
        self.l_rate = l_rate
        self.layers = []
        self.output = 0

    def append_layer(self, n_inputs, n_output, isActivate=False):
        self.layers.append(SlidingLayer(n_inputs, n_output, isActivate))

    def forward_propagate(self, input):
        self.output = input
        for layer in self.layers:
            self.output = layer.forward_propagate(self.output)
        return self.output

    def backward_propagate(self, y):
        error = self.output - y
        for layer in list(reversed(self.layers)):
            error = layer.backward_propagate(error)

    def update_weights(self):
        for layer in self.layers:
            layer.weight -= self.l_rate * layer.dw


def first_Task():
    global rbn
    global one_perceptron
    global two_perceptron
    global firstTrain
    if not firstTrain:
        messagebox.showinfo(title="ошибка", message="Сети не обучены")
        return
    x_test = np.linspace(-3, 3, 600)
    plt.title(r"Аппроксимация функции $f(x) = {e^{(\cos(x)}*sin(x))}}$")
    plt.plot(x_test, one_perceptron.predict(x_test), "b",
             label="Персептрон с одним скрытым слоем")
    plt.plot(x_test, two_perceptron.predict(x_test), "k",
             label="Персептрон с двумя скрытыми слоями")
    plt.plot(x_test, rbn.predict(x_test), "g", label="Радиально-базисная сеть")
    plt.plot(x_test, function_x(x_test), "r",
             label="Реальный график")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()


def second_Task(sequence_len, dataset, testset):
    global rnn
    global perceptron
    global secondTrain
    if not secondTrain:
        messagebox.showinfo(title="Ошибка", message="Сети не обучены")
        return
    mean = np.mean(dataset)
    std = np.std(dataset)
    rnn_y = []
    perceptron_y = []
    for i in dataset[-sequence_len:]:
        rnn_y.append((i - mean) / std)
        perceptron_y.append((i - mean) / std)
    for i in range(len(testset)):
        out_rnn = rnn.forward_propagate(
            np.array(rnn_y[i:i+sequence_len]).reshape(1, sequence_len)).reshape(-1)
        out_perceptron = perceptron.forward_propagate(
            np.array(perceptron_y[i:i+sequence_len]).reshape(1, sequence_len)).reshape(-1)
        rnn_y.append(out_rnn[0])
        perceptron_y.append(out_perceptron[0])
    plt.plot(np.array(rnn_y[sequence_len:]) * std + mean, label="Сеть Элмана")
    plt.plot(np.array(perceptron_y[sequence_len:])
             * std + mean, label="Персептрон")
    plt.plot(testset, label="Реальная температура")
    plt.xlabel("день")
    plt.ylabel("t")
    plt.legend()
    plt.grid()
    plt.show()


def load_txt(filename):
    lines = open(filename).readlines()
    values = []
    for line in lines:
        values.append(float(line.replace("\n", "")))
    return values


def train_first_Networks():
    global rbn
    global one_perceptron
    global two_perceptron
    global firstTrain
    firstTrain = True
    rbn = RBF_NeuralNetwork(5)
    x = np.random.uniform(-3, 3, size=[100000, 1])
    y = function_x(x)
    dataset = list()
    for i in range(len(x)):
        dataset.append([x[i][0], y[i][0]])
    one_perceptron = Perceptron(0.1)
    one_perceptron.append_layer(1, 7, True, True)
    one_perceptron.append_layer(7, 1, False, False)
    one_perceptron.fit(dataset)
    two_perceptron = Perceptron(0.1)
    two_perceptron.append_layer(1, 7, True, True)
    two_perceptron.append_layer(7, 5, True, True)
    two_perceptron.append_layer(5, 1, False, False)
    two_perceptron.fit(dataset)
    rbn.sigma = np.std(y)
    rbn.fit(x, y)


def train_second_Networks(sequence_len, dataset):
    global perceptron
    global rnn
    global secondTrain
    secondTrain = True
    mean = np.mean(dataset)
    std = np.std(dataset)
    rnn = RecurrentNeuralNetwork(sequence_len, 60, 1, 0.002)
    perceptron = SlidingPerceptron(0.002)
    perceptron.append_layer(sequence_len, 60, isActivate=True)
    perceptron.append_layer(60, 1)
    for epoch in range(2500):
        q = np.random.randint(0, sequence_len)
        for i in range(q, len(dataset) - sequence_len, sequence_len):
            x = (np.array(dataset[i: i+sequence_len]
                          ).reshape(1, sequence_len) - mean) / std
            y = (np.array(dataset[i+sequence_len: i +
                 sequence_len+1]).reshape(1, 1) - mean) / std
            rnn.forward_propagate(x)
            rnn.backward_propagate(y)
            rnn.update_weights()
            perceptron.forward_propagate(x)
            perceptron.backward_propagate(y)
            perceptron.update_weights()


main()
