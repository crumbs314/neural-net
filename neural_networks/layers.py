"""
Author: Sophia Sanborn, Sagnik Bhattacharya
Institution: UC Berkeley
Date: Spring 2020
Course: CS189/289A
Website: github.com/sophiaas, github.com/sagnibak
"""

import numpy as np
from abc import ABC, abstractmethod

from neural_networks.activations import initialize_activation
from neural_networks.weights import initialize_weights
from collections import OrderedDict

from typing import Callable, List, Literal, Tuple, Union


class Layer(ABC):
    """Abstract class defining the `Layer` interface."""

    def __init__(self):
        self.activation = None

        self.n_in = None
        self.n_out = None

        self.parameters = {}
        self.cache = {}
        self.gradients = {}

        super().__init__()

    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        pass

    def clear_gradients(self) -> None:
        self.cache = OrderedDict({a: [] for a, b in self.cache.items()})
        self.gradients = OrderedDict(
            {a: np.zeros_like(b) for a, b in self.gradients.items()}
        )

    def forward_with_param(
        self, param_name: str, X: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Call the `forward` method but with `param_name` as the variable with
        value `param_val`, and keep `X` fixed.
        """

        def inner_forward(param_val: np.ndarray) -> np.ndarray:
            self.parameters[param_name] = param_val
            return self.forward(X)

        return inner_forward

    def _get_parameters(self) -> List[np.ndarray]:
        return [b for a, b in self.parameters.items()]

    def _get_cache(self) -> List[np.ndarray]:
        return [b for a, b in self.cache.items()]

    def _get_gradients(self) -> List[np.ndarray]:
        return [b for a, b in self.gradients.items()]


def initialize_layer(
    name: str,
    activation: str = None,
    weight_init: str = None,
    n_out: int = None,
) -> Layer:
    """Factory function for layers."""
    return FullyConnected(
        n_out=n_out, activation=activation, weight_init=weight_init,
    )


class FullyConnected(Layer):
    """A fully-connected layer multiplies its input by a weight matrix, adds
    a bias, and then applies an activation function.
    """

    def __init__(
        self, n_out: int, activation: str, weight_init="xavier_uniform"
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.activation = initialize_activation(activation)

        # instantiate the weight initializer
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int, int]) -> None:
        """Initialize all layer parameters (weights, biases)."""
        self.n_in = X_shape[1]

        ### BEGIN YOUR CODE ###
        W = self.init_weights((self.n_in, self.n_out)) ### self.init_weights(...)
        b = np.zeros((1, self.n_out)) ### ...

        self.parameters = OrderedDict({"W": W, "b": b})
        self.cache: OrderedDict = OrderedDict()  # cache for backprop  ### ...
        self.gradients: OrderedDict = OrderedDict({"W": np.zeros((self.n_in, self.n_out)), "b": np.zeros((1, self.n_out))})  # parameter gradients initialized to zero
                                           # MUST HAVE THE SAME KEYS AS `self.parameters` ### ...
        ### END YOUR CODE ###

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: multiply by a weight matrix, add a bias, apply activation.
        Also, store all necessary intermediate results in the `cache` dictionary
        to be able to compute the backward pass.

        Parameters
        ----------
        X  input matrix of shape (batch_size, input_dim)

        Returns
        -------
        a matrix of shape (batch_size, output_dim)
        """
        # initialize layer parameters if they have not been initialized
        if self.n_in is None:
            self._init_parameters(X.shape)

        ### BEGIN YOUR CODE ###
        
        # perform an affine transformation and activation
        b = self.parameters["b"]
        b = np.repeat(b, X.shape[0], axis=0)


        W = self.parameters["W"]
        ### print(W)
        Z = np.add(np.matmul(X, W), b)

        self.cache["activation_Input"] = Z ### 2x4
        self.cache["layer_Input"] = X ### 2x3

        activation_Func = np.vectorize(self.activation.forward)
        out = activation_Func(Z)
        
        ### print(f"Y shape: {out.shape}")

        
        ### dL/dw = dL/dz * dz/dW = dL/dz * X
        # store information necessary for backprop in `self.cache`

        ### END YOUR CODE ###

        return out

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for fully connected layer.
        Compute the gradients of the loss with respect to:
            1. the weights of this layer (mutate the `gradients` dictionary)
            2. the bias of this layer (mutate the `gradients` dictionary)
            3. the input of this layer (return this)

        Parameters
        ----------
        dLdY  derivative of the loss with respect to the output of this layer
              shape (batch_size, output_dim)

        Returns
        -------
        derivative of the loss with respect to the input of this layer
        shape (batch_size, input_dim)
        """
        ### BEGIN YOUR CODE ###
        
        # unpack the cache
        Z = self.cache["activation_Input"]
        X = self.cache["layer_Input"]

        deri_Loss_WRT_Z = self.activation.backward(Z, dLdY)
        ### print(f"deri_Loss_WRT_Z shape should be 2x4: {deri_Loss_WRT_Z.shape}")

        W = self.parameters["W"]

        ### print(f"W shape: {W.shape}")

        ### print(f"X shape: {X.shape}")

        dX = np.matmul(deri_Loss_WRT_Z, np.transpose(W))

        deri_Loss_WRT_W = np.transpose(np.matmul(np.transpose(deri_Loss_WRT_Z), X))
        ### print(f"deri_Loss_WRT_W shape should be 3,4: {deri_Loss_WRT_W.shape}")
        ### print(f"deri_Loss_WRT_Z shape should be 2,4: {deri_Loss_WRT_Z.shape}")
        ### print(f"dW shape should be 2,3: {X.shape}")

        self.gradients["W"] = deri_Loss_WRT_W

        deri_loss_WRT_Z_summed = np.sum(deri_Loss_WRT_Z, axis = 0)
        self.gradients["b"] = deri_loss_WRT_Z_summed

        # compute the gradients of the loss w.r.t. all parameters as well as the
        # input of the layer

        # store the gradients in `self.gradients`
        # the gradient for self.parameters["W"] should be stored in
        # self.gradients["W"], etc.

        ### END YOUR CODE ###

        return dX
