# Experience replay buffer for off policy RL algorithm
from typing import Tuple
import numpy as np 


class ReplayBuffer:
  # Fixed size cirular replay buffer storing transitions as numpy arrays


  def __init__(
      self,
      capacity: int,
      observation_dim: int ,
      seed: int = 676767,
  ) -> None:
    """
      capacity: Maximum number of transitions to store.
      observation_dim: Dimensionality of observation vectors.
      seed: random seed for sampling
    
    """

    self._capacity = capacity
    self._size = 0
    self._position = 0
    self._rng = np.random.RandomState(seed)


    self._observation = np.zeros((capacity, observation_dim), dtype=np.float32)
    self._actions = np.zeros(capacity, dtype=np.int64)
    self._rewards = np.zeros(capacity, dtype=np.float32)
    self._next_observations = np.zeros(
      (capacity, observation_dim), dtype=np.float32  
    )
    self._dones = np.zeros(capacity, dtype=np.float32)



  @property
  def size(self) -> int:
    return self._size # returns the number of stored transitions
  

  def add(
      self,
      observation: np.ndarray,
      action: int,
      reward: float,
      next_observation: np.ndarray,
      done: bool,
  ) -> None:
    """
    Adds a transition to the buffer

    observation: current state observation.
    action: Action taken
    reward: Reward received.
    next_observation: Next state observation
    done: Whether the episode ended.
    """

    idx = self._position
    self._observation[idx] = observation
    self._actions[idx] = action
    self._rewards[idx] = reward
    self._next_observations[idx] = next_observation
    self._dones[idx] = float(done)


    self._position = (self._position + 1)%self._capacity
    self._size = min(self._size + 1, self._capacity)
  


  def sample(
      self, 
      batch_size: int 
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample a random batch of transitions.
    
    batch_size: Number of transitions to sample.

    Tuple of (observation, actions, rewards, next_observations, dones)

    ValueError if batch_size exceeds buffer size.
    
    """

    if batch_size > self._size:
      raise ValueError(
        f"Cannot sample {batch_size} from buffer of size {self._size}"
      )
    
    indices = self._rng.choice(self._size, size=batch_size, replace=False)
    return (
      self._observation[indices],
      self._actions[indices],
      self._rewards[indices],
      self._next_observations[indices],
      self._dones[indices]
    )
  
  def can_sample(self, batch_size: int) -> bool:
    # Check if the buffer has enough transition for a batch.
    return self._size >= batch_size