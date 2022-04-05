from Network import Network, unit_vector
import pickle
import numpy as np


class NNModel:

    """
    A model for a basic neural network with implemented
    fitting and prediction functionality.
    """

    def __init__(self, N_array: np.ndarray):
        self.N_array = N_array
        self.Network = Network(N_array)


    def fit(self, X, y, bin_size: int = 10, epochs: int = 5, learn_rate: float = 0.01) -> None:
        """
        Trains a neural network with the given data
        X: ndarray of training data equal to the input
        y: Response data to X
        bin_size: Chunk size for bin trainings
        epochs: Number of times to iterate the data
        """
        assert(len(X) == len(y))

        if not ( isinstance(y[0], np.ndarray) ):
            y = vectorize_integer(y)

        N = len(X)
        ind_array = np.arange(N)
        for iter in range(epochs):
            for ind in np.random.choice(ind_array, size = (int(N/bin_size), bin_size), replace = False):
                self.Network.train_single_bin(X[ind], y[ind], learn_rate)
            print(f"Finished training iteration {iter+1}")

    
    def predict(self, X: np.ndarray, normalize: bool = False, only_max_ind: bool = False) -> np.ndarray:
        """
        Given a single instance of x-data, predicts
        the response variable y.
        normalize: Normalizes each vector y_pred in the 1-norm
        only_max:  Sets all outputs in y to zero except the maximum
        """
        y_pred = np.zeros((len(X), self.N_array[-1]))
        for i, x in enumerate(X):
            self.Network.update_network(x)
            y_pred[i] = self.Network.get_output()

            if normalize: y_pred[i] /= np.linalg.norm(y_pred[i], ord = 1)
            if only_max_ind: y_pred[i] = unit_vector(np.argmax(y_pred[i]), self.N_array[-1])

        return y_pred
                

    def save(self, filename: str = "Network.pickle") -> None:
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    
    @staticmethod
    def load(filename: str = "Network.pickle"):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model


def evaluate_accuracy(y_pred: np.ndarray, y: np.ndarray):
    """
    Given predicted data and test response date,
    evaluates the accuracy of a model.
    """
    assert(len(y_pred) == len(y))

    amt_correct = 0
    N = len(y_pred)
    for y_hat, y in zip(y_pred, y):
        amt_correct += np.all(y_hat == y)
    return amt_correct/N


def vectorize_matrices(X) -> np.ndarray:
    """
    Given some list of matrices, flattens each matrix
    and returns as a numpy ndarray.
    """
    X_vectorized = np.zeros( (len(X), np.prod(np.array(X[0]).shape)) )
    for i, x in enumerate(X):
        X_vectorized[i] = np.array(x).flatten()
    return X_vectorized


def vectorize_integer(y) -> np.ndarray:
    """
    Given some list of integers, turns each element
    into unit vectors with 1 in the given position.
    """
    dim = np.max(y) + 1
    y_vectorized = np.zeros( (len(y), dim) )
    for i, y_el in enumerate(y):
        y_vectorized[i] = unit_vector(y_el, dim)
    return y_vectorized