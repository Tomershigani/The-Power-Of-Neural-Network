# Unleashing the Power of Neural Networks: From MNIST Classification to Overfitting

This project consists of two parts that demonstrate the capabilities and limitations of neural networks.

In the first part, I built a classifier for the MNIST dataset from scratch using PyTorch. I did not use any built-in tensor
functions and implemented the layers, activation functions, loss functions, differentiation (backward), and optimization myself.
Through this process, I achieved an accuracy of 92% on the MNIST dataset.

In the second part, I demonstrated the phenomenon of overfitting to random labels. Using the MNIST dataset, I trained the network
on the first 128 samples from the training dataset, with the parameters set to shuffle=False and a batch size of 128. I then generated
random labels from a Bernoulli distribution with a probability of ½, effectively assigning each sample a random label of either 0 or 1.
Despite the lack of meaningful connections between the data and the labels, I showed that the network I implemented was able to
achieve a loss value of ~0 (using cross-entropy loss) on the training set and an accuracy of 50% on the test set. This large gap between the
training set and the test set demonstrates the power of neural networks to learn almost anything, even random data with no real connections.

Overall, this project highlights both the impressive capabilities and the limitations of neural networks, and emphasizes the importance of
carefully considering the data and the task at hand when building and training these models.

<img src="https://user-images.githubusercontent.com/81327428/221197725-95ced2ed-8215-4496-b0e2-4fc90763ce11.png" width="100" height="100">      <img src="https://user-images.githubusercontent.com/81327428/221198266-d3344801-342c-4c8e-a6d1-730bc0daf5f1.png" width="100" height="100">     <img src="https://user-images.githubusercontent.com/81327428/221198741-f42406b8-c4da-4b49-ba15-0d7dee31544a.png" width="100" height="100">
