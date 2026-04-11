# Abstract base class for control agents.

from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
  # Interface that all control agents must implement

  @abstractmethod
  def select_action(self, observation: np.ndarray) -> int:
    """
    Select an action given the current observation.


    Args: 
      observation: Environment observation vector
    Returns:
      Discrete action index.
    """


  @abstractmethod
  def reset(self) -> None:
    """Reset agent internal state for a new episode"""