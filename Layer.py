import numpy as np


class Layer:
    
    """
    Layer of a neural network containing entire vector of values,
    matrix of weights and bias.
    """

    def __init__(self, N: int, N_suc: int, seed: int = None):
        if seed: np.random.seed(seed)                         # Seed for generation of weights and biases
        self.__N = N                                          # Amount of neurons in the given layer
        self.__neurons = np.zeros(N)                          # Values of each neuron
        self.__weights = np.random.uniform(-1, 1, (N_suc, N)) # Matrix of weights to succeeding layer
        self.__bias = np.random.uniform(-1, 1, N_suc)         # Bias to succeeding layer
        self.__prev: Layer = None                             # Pointers to preceeding and succeeding layers
        self.__next: Layer = None


    def set_values(self, values: np.ndarray):
        assert( len(values) == self.__N )
        self.__neurons = values

    def calculate_next_layer(self) -> np.ndarray:
        return Layer.sigmoid( self.__weights@self.__neurons + self.__bias )

    def set_prev(self, prev) -> None:
        self.__prev = prev

    def get_prev(self):
        return self.__prev

    def set_next(self, next) -> None:
        self.__next = next

    def get_next(self):
        return self.__next

    
    def update_weights(self, d_weight: np.ndarray) -> None:
        self.__weights += d_weight

    
    def update_bias(self, d_bias: np.ndarray) -> None:
        self.__bias += d_bias


    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1/(1+np.exp(-x))

    
    @staticmethod
    def sigmoid_prime(x: np.ndarray) -> np.ndarray:
        sigmoid = Layer.sigmoid(x)
        return sigmoid*(1-sigmoid)

    
    def phi(self):
        """
        A term included in the backprop algorithm
        (partial derivative of aL wrt. zL)
        """
        return Layer.sigmoid_prime(self.__weights@self.__neurons + self.__bias)
        

    @property
    def neurons(self):
        return self.__neurons


    @property
    def weights(self):
        return self.__weights


    @property
    def bias(self):
        return self.__bias