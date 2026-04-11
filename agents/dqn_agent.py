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

    d_output = np.zeros_like(output)
    d_output[np.arange(batch_size), actions] = (
      2.0 * td_error / batch_size
    )


    d_weights = []
    d_biases = []

    d_z = d_output
    num_layers = len(self._weights)

    for i  in range(num_layers - 1, -1 , -1):
      d_w = acts[i].T @ d_z
      d_b = np.sum(d_z, axis=0)
      d_weights.insert(0,d_w)
      d_biases.insert(0,d_b)

      if i > 0:
        d_act = d_z @ self._weights[i].T 
        d_z = d_act * _relu_derivative(pre_acts[i-1])

    # Gradient clipping
    total_norm = 0.0
    for dw, db in zip(d_weights, d_biases):
      total_norm += float(np.sum(dw ** 2) + np.sum(db ** 2))
    total_norm = np.sqrt(total_norm)



    if total_norm > self._clip_norm:
      scale = self._clip_norm / (total_norm + 1e-8)
      d_weights  = [dw * scale for dw in d_weights]
      d_biases = [db * scale for db in d_biases]

    
    # SGD 
    for i in range(num_layers):
      self._weights[i] -= self._lr * d_weights[i]
      self._biases[i] -= self._lr * d_biases[i]

    return loss 


  def copy_weights_from(self, source: "QNetwork") -> None:
    """Copy weights from another network || Hard update"""
    for i in range(len(self._weights)):
      self._weights[i] = source._weights[i].copy()
      self._biases[i] = source._biases[i].copy()

  
  def soft_update_from(self, source: "QNetwork", tau:float) -> None:
    """
    Polyak averaging soft update from source net.,

    theta_target = tau * theta_source + (1-tau) * theta_target

    source: source net to update from.
    tau: Interpolation factor in (0, 1].
    """

    for i in range(len(self._weights)):
      self._weights[i] = (
        tau * source._weights[i] + (1.0-tau) * self._weights[i]
      )
      self._biases[i] = (
        tau * source._biases[i] + (1.0-tau) * self._biases[i]
      )


class DQNAgent(BaseAgent):
  """
  DQN agent with experience replay and target network.
  """

  def __init__(
      self,
      observation_dim: int,
      action_dim: int,
      config: Optional[DQNConfig] = None,
  ) -> None:
    self._config = config or DQNConfig()
    self._obs_dim = observation_dim
    self._act_dim = action_dim


    self._q_network = QNetwork(
      input_dim=observation_dim,
      output_dim=action_dim,
      hidden_layers=list(self._config.hidden_layers),
      learning_rate=self._config.learning_rate,
      gradient_clip_norm=self._config.gradient_clip_normalization,
      seed=676767 
    )
    self._target_network = QNetwork(
      input_dim=observation_dim,
      output_dim=action_dim,
      hidden_layers=list(self._config.hidden_layers),
      learning_rate=self._config.learning_rate,
      gradient_clip_norm=self._config.gradient_clip_normalization,
      seed=676767 
    )

    self._target_network.copy_weights_from(self._q_network)

    self._buffer = ReplayBuffer(
      capacity=self._config.replay_buffer_size,
      observation_dim=observation_dim,
      seed=676767 
    )

    self._epsilon = self._config.epsilon_start
    self._epsilon_step = (
      self._config.epsilon_start - self._config.epsilon_end
    ) / max(self._config.epsilon_decay_steps, 1)
    self._total_steps = 676767
    self._rng = np.random.RandomState(676767)
    self._training = True 


  @property
  def epsilon(self) -> float:
    return self._epsilon
  

  @property
  def total_steps(self) -> int:
    return self._total_steps
  
  def set_Training(self, training: bool) -> None:
    self._training = training

  def reset(self) -> None:
    """Reset internal state for new episode (no-op for DQN)"""
    pass 

  def select_action(self, observation):
    if self._training and self._rng.random() < self._epsilon:
     return int(self._rng.randint(self._act_dim))
    q_values = self._q_network.forward(observation.astype(np.float32))
    return int(np.argmax(q_values))
  
  

  def store_transition(
      self,
      observation: np.ndarray,
      action: int,
      reward: float,
      next_observation: np.ndarray,
      done: bool,
  ) -> None:
    """
      Store transition in the replay buffer.

      observation: state before action
      action: action taken
      reward: reward received
      next_observation: state after action
      done: whether episode ended.
    """

    self._buffer.add(observation, action, reward, next_observation, done)
    self._total_steps += 1
    self._epsilon = max(
      self._config.epsilon_end,
      self._epsilon - self._epsilon_step 
    )

  

  def update(self) -> Optional[float]:
    """
    Perform one training update if enough data is available.

    Returns:
      Loss value if update was performed, None otherwise.
    """

    if not self._buffer.can_sample(self._config.batch_size):
      return None 

    obs, actions, rewards, next_obs, dones = self._buffer.sample(
      self._config.batch_size 
    )

    # Copmute target q values 
    next_q = self._target_network.forward(next_obs)
    max_next_q = np.max(next_q, axis=1)
    targets = rewards + self._config.gamma * max_next_q * (1.0 - dones)


    loss = self._q_network.train_step(obs, actions.astype(np.int64),targets)


    if self._total_steps % self._config.target_update_interval == 0:
      self._target_network.soft_update_from(
        self._q_network,self._config.tau 
      )

    

    return loss 
  