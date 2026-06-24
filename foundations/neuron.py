import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        z = np.dot(x, w) + b
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        if(activation == "sigmoid"):
            ans = 1/(1+np.exp(-z))
        else:
            ans = np.maximum(0, z)

        return np.round(ans, 5)
        # return round(your_answer, 5)
        pass
