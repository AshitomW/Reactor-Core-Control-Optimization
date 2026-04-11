# DQN agent implementation based on my understanding, might be incorrect or incomplete

from typing import List, Optional
import numpy as np

from config import DQNConfig
from agents.abstract import BaseAgent
from agents.replay_buffer import ReplayBuffer



def _relu(x: np.ndarray) -> np.ndarray:
  # RELU activation function
  return np.maximum(0.0,x)


def _relu_derivative(x: np.ndarray) -> np.ndarray:
  # Derivative of RELU
  return (x>0).astype(np.float32)



class QNetwork:
  """
  Fully connected Q-Network implemented using NUMPY. [READ DISCLAIMER IN THE FILE FOR CAUTION]

  Uses Relu activation for hidden layers and linear output.
  Trained via SGD with graident clipping
  
  """



  def __init__(
      self,
      input_dim: int ,
      output_dim: int ,
      hidden_layers: List[int],
      learning_rate: float,
      gradient_clip_norm: float,
      seed: int = 676767
  ) -> None:
    """
    input_dim: Obervation vector dimensionality.
    output_dim: Number of actions (Q-value outputs)
    hidden_layers: List of hidden layer sizes.
    learning_rate: SGD learning rate.
    gradient_clip_norm: Maximum gradient norm for clipping.
    seed: Random seed for weight intialization.
    
    """

    self._lr = learning_rate
    self._clip_norm = gradient_clip_norm
    rng = np.random.RandomState(seed)


    layer_sizes = [input_dim] + hidden_layers + [output_dim]
    self._weights: List[np.ndarray] = []
    self._biases: List[np.ndarray] = []


    for i in range(len(layer_sizes) - 1):
      fan_in = layer_sizes[i]
      fan_out = layer_sizes[i + 1]

      # HE intialization
      std = np.sqrt(2.0/fan_in)
      self._weights.append(
        rng.randn(fan_in, fan_out).astype(np.float32) * std 
      )
      self._biases.append(np.zeros(fan_out, dtype=np.float32))
  

  def forward(self, x: np.ndarray) -> np.ndarray:
    """
      Forward pass through the network.

      x: input array of shape (batch, iput_dim) or (input_dim, ).

      Will Return, Q-Values of shape (batch, output_dim) or (output_dim,)
    """

    single = x.ndim == 1
    if single:
      x = x.reshape(1, -1)
    activations = x 

    for i in range(len(self._weights) - 1 ):
      z = activations @ self._weights[i] + self._biases[i]
      activations = _relu(z)

    output = activations @ self._weights[-1] + self._biases[-1]

    if single:
      return output.squeeze(0)
    return output
  


  def forward_with_cache(
      self,
      x: np.ndarray
  ) -> tuple:
    
    """
    Forward pass that will cache intermediate values for backpropagation

    x: input array of shape (batch, input_dim).

    returns: tuple of (output, list_of_pre_activations, list_of_activations)
    """


    pre_acts = []
    acts = [x]

    activations = x
    for i in range(len(self._weights) - 1 ):
      z = activations @ self._weights[i] + self._biases[i]
      pre_acts.append(z)
      activations = _relu(z)
      acts.append(activations)

    z_out = activations @ self._weights[-1] + self._biases[-1]
    pre_acts.append(z_out)
    acts.append(z_out)

    return z_out, pre_acts, acts 
  
  def train_step(
      self,
      observations: np.ndarray,
      actions: np.ndarray,
      targets: np.ndarray
  ) -> float:
    
    """
      Perform one gradient descent step.
    
      observations: Batch of observations (batch, input_dim)
      actions: batch of action indices (batch,).
      targets: Target Q values for the selected actions (batch, ).

      Mean Squared Loss Value will be returned.
    """

    batch_size = observations.shape[0]
    output, pre_acts , acts = self.forward_with_cache(observations)

    # compute loss only for selected actions
    q_selected = output[np.arange(batch_size), actions]
    td_error = q_selected - targets
    loss = float(np.mean(td_error ** 2))


    # Backpropagation (WIP)


