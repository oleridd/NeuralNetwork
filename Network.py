import numpy as np
from Layer import Layer


class Network:

    """
    A neural network defined by N_array. The network will have as many layers as the
    length of N_array, and each layer will have the dimension of the respective
    integer at the coresponding index in the array. N_array also includes the output
    dimesion as the last value.
    """

    def __init__(self, N_array: np.ndarray):
        self.__N0 = N_array[0]       # Final value is the dimension of output
        self.__out_dim = N_array[-1]
        self.__M = len(N_array)-1    # Amount of layers in the network excluding output
        self.__layers = [Layer(N_array[i], N_array[i+1]) for i in range(self.__M)]
        for i in range(1, self.__M): # Setting pointers to individual layers
            self.__layers[i].set_prev(self.__layers[i-1])
            self.__layers[i-1].set_next(self.__layers[i])


    def __set_input(self, network_input: np.ndarray) -> None:
        """
        Sets the input values for the network
        """
        assert( len(network_input) == self.__N0 )
        self.__layers[0].set_values(network_input)
    

    def update_network(self, input: np.ndarray) -> None:
        """
        Updates entire network based on input
        """
        self.__set_input(input)
        for i in range(1, self.__M):
            self.__layers[i].set_values( self.__layers[i-1].calculate_next_layer() )
    

    def get_output(self) -> np.ndarray:
        return self.__layers[-1].calculate_next_layer()


    def get_cost(self, input: np.ndarray, ans: int) -> float:
        """
        Given an input from the training data and a correct answer,
        calculates the cost function of the given input.
        ans: index of correct state
        """
        self.update_network(input)
        out = self.get_output()
        cost = np.sum([(out[i] - 1)**2 if i == ans else out[i]**2 for i in range(self.__out_dim)])
        return cost

    
    def __get_empty_arrays(self) -> tuple:
        """
        Returns empty arrays for weights and biases
        """
        d_weights = [np.zeros(layer.weights.shape) for layer in self.__layers]
        d_bias  = [np.zeros(layer.bias.shape) for layer in self.__layers]
        return d_weights, d_bias


    def __backprop_tail(self, res: int, last_layer: Layer, output: np.ndarray) -> tuple:
        """
        Performs backpropagation on the last two layers
        of the network.
        """
        d_bias = last_layer.phi()*2*(output - res)
        d_weights = np.transpose(np.outer(last_layer.neurons, d_bias))
        d_prev = last_layer.weights.transpose()@d_bias
        return d_weights, d_bias, d_prev


    def __backprop_inner(self, layer: Layer, d_prev: np.ndarray, is_first: bool = False) -> tuple:
        """
        Performs backpropagation on the inner layers of the network.
        is_first: Whether the given layer is the first one
        """
        d_bias = layer.phi()*d_prev
        d_weights = np.transpose(np.outer(layer.neurons, d_bias))
        if not is_first:
            d_prev = layer.weights.transpose()@d_bias
        else:
            d_prev = None
        return d_weights, d_bias, d_prev


    def __backprop(self, train_sample: np.ndarray, res: int) -> tuple:
        """
        Preforms backpropagation on the network and returns the gradient
        for a given training_sample.
        train_sample: Sample to be trained
        res: Actual solution for the training sample (as an index)
        """
        l = self.__layers # Making it a bit easier to call layers
        self.update_network(train_sample)
        output = self.get_output()

        d_weights, d_bias = self.__get_empty_arrays()

        d_weights[-1], d_bias[-1], d_prev = self.__backprop_tail(res, l[-1], output) # Handling last layer

        # Iterating through layers:
        for i in range(self.__M-2, -1, -1):
            d_weights[i], d_bias[i], d_prev = self.__backprop_inner(l[i], d_prev, i == 0)

        return d_weights, d_bias
    

    def train_single_bin(self, bin, res, learn_rate) -> None:
        """
        Given a bin of training examples and corresponding results,
        trains the network for a single iteration of gradient descent.
        """
        # Backpropagating for all bins
        k = len(bin) # Amount of examples in bin
        d_weights, d_bias = self.__get_empty_arrays()
        for i in range(k):
            dw, db = self.__backprop(bin[i], res[i])
            for i in range(self.__M):
                d_weights[i] += dw[i]/k
                d_bias[i] += db[i]/k
        
        # Updating network parameters:
        for i, layer in enumerate(self.__layers):
            layer.update_weights(-d_weights[i]*learn_rate)
            layer.update_bias(-d_bias[i]*learn_rate)


def unit_vector(ind: int, dim: int):
    vec = np.zeros(dim)
    vec[ind] = 1
    return vec