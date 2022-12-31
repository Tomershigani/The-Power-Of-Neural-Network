

def evaluate_hw1():
    import torch
    from mlxtend.data import mnist_data
    import torchvision.datasets as dsets
    import torchvision.transforms as transforms
    import numpy as np

    def make_p(y):

        p = torch.argmax(y, 1)
        return p

    def row_softmax(a):
        max_a = torch.max(a).item()
        a = a - max_a
        return torch.exp(a) / sum(torch.exp(a))

    def softmax(t):
        A = torch.zeros(t.shape)
        for index, item in enumerate(t):
            A[index] = row_softmax(item)
        return A

    def tanh(t):
        return torch.div(torch.exp(t) - torch.exp(-t), torch.exp(t) + torch.exp(-t))

    def Relu(t):
        t[t < 0] = 0
        return t

    def ReluPrime(t):
        t[t < 0] = 0
        t[t > 0] = 1
        return t

    def tanhPrime(t):
        # derivative of tanh
        # t: tanh output
        return 1 - t * t

    def tanhPrime(t):
        # derivative of tanh
        # t: tanh output
        return 1 - t * t

    class Neural_Network:
        def __init__(self, input_size=784, output_size=10, hidden_size=100):
            # parameters
            self.inputSize = input_size
            self.outputSize = output_size
            self.hiddenSize = hidden_size

            # weights
            self.W1 = torch.randn(self.inputSize, self.hiddenSize)  # (3x2) weight matrix from input to hidden layer
            self.b1 = torch.zeros(self.hiddenSize)
            self.W2 = torch.randn(self.hiddenSize, self.outputSize)  # dtype=torch.float64
            self.b2 = torch.zeros(self.outputSize)

        def forward(self, X):
            self.z1 = torch.matmul(X, self.W1) + self.b1
            self.h = Relu(self.z1)
            self.z2 = torch.matmul(self.h, self.W2) + self.b2
            return softmax(self.z2)

        def one_hot(self, Y, size):
            one_hot_Y = torch.zeros((Y.size(0), size))
            for i, val in enumerate(Y):
                one_hot_Y[i][int(val)] = 1
            return one_hot_Y

        def set_weight(self, weights):
            self.W1 = weights["w1"]
            self.W2 = weights["w2"]
            self.b1 = weights["b1"]
            self.b2 = weights["b2"]

        def save_weights(self):
            torch.save({"w1": self.W1, "w2": self.W2, "b1": self.b1, "b2": self.b2}, "w_q1.pkl")

        def backward(self, X, y, y_hat, lr=0.1):
            batch_size = y.size(0)
            one_hot_Y = self.one_hot(y, self.outputSize)
            dl_dz2 = (1 / batch_size) * (y_hat - one_hot_Y)
            dl_dh = torch.matmul(dl_dz2, torch.t(self.W2))
            dl_dz1 = dl_dh * ReluPrime(self.h)

            self.W1 -= lr * torch.matmul(torch.t(X), dl_dz1)
            self.b1 -= lr * torch.matmul(torch.t(dl_dz1), torch.ones(batch_size))
            self.W2 -= lr * torch.matmul(torch.t(self.h), dl_dz2)
            self.b2 -= lr * torch.matmul(torch.t(dl_dz2), torch.ones(batch_size))

    # Press the green button in the gutter to run the script.
    if __name__ == '__main__':
        num_epochs = 3
        batch_size = 100
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
                                                  batch_size=batch_size,
                                                  shuffle=False)

        model = Neural_Network(784, 10, 100)
        weights = torch.load("w_q1.pkl")
        model.set_weight(weights)
        test_accuracy = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.view(-1, 784)
            y_hat = model.forward(images)
            prediction = make_p(y_hat)
            test_accuracy += (prediction == labels).sum()
        test_accuracy = np.float16(test_accuracy / 10000)
        print("Test Accuracy is: %5.10f" % (test_accuracy))

evaluate_hw1()
