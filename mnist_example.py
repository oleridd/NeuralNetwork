import mnist
import numpy as np
import NeuralNetwork as NN


def train(bin_size, epochs) -> None:
    X_train, y_train = NN.vectorize_matrices(mnist.train_images()), mnist.train_labels()
    # Constructing a neural network with two hidden layers of size 64:
    my_network = NN.NNModel(N_array = np.array([len(X_train[0]), 64, 64, np.max(y_train)+1]))
    # Training and saving the network:
    my_network.fit(X_train, y_train, bin_size, epochs)
    my_network.save(filename="mnist_example_trained.pickle")


def test() -> float:
    X_test, y_test  = NN.vectorize_matrices(mnist.test_images()), mnist.test_labels()
    # Loading a trained network:
    my_network = NN.NNModel.load("mnist_example_trained.pickle")
    y_pred = np.argmax(my_network.predict(X_test, only_max_ind=True), axis=1)
    return NN.evaluate_accuracy(y_pred, y_test)


def main() -> None:
    # train(bin_size=10, epochs=25)
    accuracy = test()
    print(f"Accuracy of network: {accuracy}")


if __name__ == "__main__":
    main()