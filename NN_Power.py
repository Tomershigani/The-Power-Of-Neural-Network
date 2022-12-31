

def evaluate_hw1():
    import torch
    import torch.nn as nn
    from mlxtend.data import mnist_data
    import torchvision.datasets as dsets
    import torchvision.transforms as transforms
    import numpy as np

    def CrossEntropy(y_pred, y_true):
        y_pred = y_pred.numpy()
        y_true = y_true.numpy()
        res = 0
        for i, j in zip(y_true, y_pred):
            if (i[0] == 1 and j[0] != 0):
                res = res - np.log(j[0])
            elif (i[0] == 0 and j[0] != 1):
                res = res - np.log(1 - j[0])
        num_of_samples = y_pred.shape[0]
        mean_bce_loss = res / num_of_samples
        return mean_bce_loss

    def make_p(y):

        p = torch.argmax(y, 1)
        return p

    def sigmoid(s):
        return 1 / (1 + torch.exp(-s))

    def sigmoidPrime(s):
        # derivative of sigmoid
        # s: sigmoid output
        return s * (1 - s)

    def tanhPrime(t):
        # derivative of tanh
        # t: tanh output
        return 1 - t * t

    def ReLU_deriv(Z):
        return Z > 0

    class Neural_Network:
        def __init__(self, input_size=784, output_size=1, hidden_size=200):
            # parameters
            self.inputSize = input_size
            self.outputSize = output_size
            self.hiddenSize = hidden_size

            # weights
            self.W1 = torch.randn(self.inputSize, self.hiddenSize)
            self.b1 = torch.zeros(self.hiddenSize)

            self.W2 = torch.randn(self.hiddenSize, self.outputSize)
            self.b2 = torch.zeros(self.outputSize)

        def set_weight(self, weights):
            self.W1 = weights["w1"]
            self.W2 = weights["w2"]
            self.b1 = weights["b1"]
            self.b2 = weights["b2"]

        def save_weights(self):
            torch.save({"w1": self.W1, "w2": self.W2, "b1": self.b1, "b2": self.b2}, "w_q1.pkl")

        def forward(self, X):
            self.z1 = torch.matmul(X, self.W1) + self.b1
            relu = nn.ReLU()
            tanh_func = nn.Tanh()
            self.h = tanh_func(self.z1)
            self.z2 = torch.matmul(self.h, self.W2) + self.b2
            return sigmoid(self.z2)

        def backward(self, X, y, y_hat, lr=.1):
            batch_size = y.size(0)
            dl_dz2 = (1 / batch_size) * (y_hat - y)

            dl_dh = torch.matmul(dl_dz2, torch.t(self.W2))
            dl_dz1 = dl_dh * tanhPrime(self.h)

            self.W1 -= lr * torch.matmul(torch.t(X), dl_dz1)
            self.b1 -= lr * torch.matmul(torch.t(dl_dz1), torch.ones(batch_size))
            self.W2 -= lr * torch.matmul(torch.t(self.h), dl_dz2)
            self.b2 -= lr * torch.matmul(torch.t(dl_dz2), torch.ones(batch_size))
            return self.W1, self.W2

    # Press the green button in the gutter to run the script.
    if __name__ == '__main__':
        num_epochs = 3
        batch_size = 128
        learning_rate = 0.1
        x, y = mnist_data()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.1307, 0.3081)
        ])


        test_dataset = dsets.MNIST(root='./data/',
                                   train=False,
                                   transform=transform,
                                   download=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=10000,
                                                  shuffle=False)



        test_features, test_labels = next(iter(test_loader))
        test_labels = torch.empty(10000, 1)
        test_features = test_features.view(-1, 784)
        test_labels = torch.bernoulli(test_labels, 0.5)

        model = Neural_Network(784, 1, 100)
        weights = torch.load("w_q1.pkl")
        model.set_weight(weights)
        y_hat = model.forward(test_features)
        prediction = torch.round(y_hat)
        accuracy = (prediction == test_labels).sum()
        accuracy = np.float16(accuracy / 10000)
        cp = CrossEntropy(y_hat, test_labels)
        print("Test Cross-Entropy loss: ", cp)
        print("Test Accuracy is: %5.10f" % (accuracy))

evaluate_hw1()

