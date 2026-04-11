# GYM Compatible reactor control environment
import math
from typing import Any, Dict, Optional, Tuple
import numpy as np


from config import (
  EnvironmentConfig,
  PhysicsConstant,
  RewardConfig
)

from env.reward import RewardCalculator
from physics.point_kinetics import PointKineticsModel
from physics.thermal_model import ThermalModel
from physics.constants import ReactorState



class ReactorEnv:
  """
  OpenAI Gym style environment for nuclear reactor control.

  State Vector: [flux_error, flux_rate, tmeperature_deviation,
  precursor_ratio, external_reactivity]
  

  Action: discrete index mapping to reactivity insertion steps.
  
  """


  def __init__(self, env_config: Optional[EnvironmentConfig] = None , physics_constants: Optional[PhysicsConstant] = None, reward_config: Optional[RewardConfig] = None):
    self._env_cfg = env_config or EnvironmentConfig()
    self._phys_cfg = physics_constants or PhysicsConstant()
    self._reward_cfg = reward_config or RewardConfig()


    self._kinetics = PointKineticsModel(self._phys_cfg)
    self._thermal = ThermalModel(self._phys_cfg)
    self._reward_calc = RewardCalculator(self._reward_cfg)

    self._rng = np.random.RandomState(self._env_cfg.seed)

    self._action_map = self._build_action_map()
    self.observation_dim = 5 
    self.action_dim = self._env.cfg.num_actions



    self._flux: float = 0.0
    self._precursor: float = 0.0
    self._temperature: float = 0.0
    self._external_reactivity: float = 0.0
    self._prev_flux: float = 0.0
    self._step_count: int = 0
    self._target_flux: float = self._env_cfg.target_flux
    self._done: bool = True 


  def _build_action_map(self) -> Dict[int,float]:
    # create mapping from discrete action indices to reactivity steps.

    n = self._env_cfg.num_actions
    half = n // 2
    action_map = {}
    for i in range(n):
      action_map[i] = (i-half) * self._env_cfg.reactivity_steps
    return action_map
  

  @property
  def target_flux(self) -> float:
    # Current target flux setpoint
    return self._target_flux
  

  @target_flux.setter
  def target_flux(self,value:float):
    if value <= 0:
      raise ValueError(f"Target flux must be positivve, got {value}")
    self._target_flux = value


  def seed(self, seed: int) -> None:
    self._rng = np.random.RandomState(seed)

  
  def reset(self) -> np.ndarray:
    # Resetting the environment to steady state intial conditions

    self._flux = self._env_cfg.nominal_flux
    self._precursor = self._kinetics.equilibrium_precursor(self._flux)
    self._temperature = self._thermal.equilibrium_temperature(self._flux)
    self._external_reactivity = 0.0
    self._prev_flux = self._flux
    self._step_count = 0
    self._target_flux = self._env_cfg.target_flux
    self._done = False 


    return self._get_observation()
  
  def step(
      self, 
      action: int 
  ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
    
    """
    Execute one control step 
    """

    if self._done:
      raise RuntimeError(
        "Episode has terminated. Call reset() before step()."
      )
    
    if action not in self._action_map:
      raise ValueError(
        f"Invalid action {action}. Valid range: 0..{self.action_dim-1}"
      )
    

    reactivity_change = self._action_map[action]
    self._external_reactivity += reactivity_change


    # Adding stochastic disturbance if enabled.

    disturbance = self._sample_disturbance()
    effective_external = self._external_reactivity + disturbance

    temp_feedback = self._thermal.compute_reactivity_feedback(
      self._temperature
    )


    total_reactivity = effective_external + temp_feedback



    self._prev_flux = self._flux
    self._flux, self._precursor = self._kinetics.step(
      self._flux, self._precursor, total_reactivity, self._env_cfg.dt 
    )
    self._temperature = self._thermal.step(
      self._temperature, self._flux, self._env_cfg.dt 
    )

    self._step_count += 1


    is_violation = self._check_safety_violation()
    time_limit = self._step_count >= self._env_cfg.max_steps


    self._done = is_violation or time_limit


    reward = self._reward_calc.compute(
      flux=self._flux,
      target_flux=self._target_flux,
      fuel_temperature=self._temperature,
      nominal_temperature=self._phys_cfg.nominal_temperature,
      reactivity_change=abs(reactivity_change),
      is_safety_violation=is_violation, 
    )

    info = self._build_info(
      total_reactivity, temp_feedback, is_violation, disturbance
    )


    return self._get_observation(), reward, is_violation, disturbance


  def get_reactor_state(self) -> ReactorState:
    """
    Return a snapshot of the current reactor physical state.
    """

    temp_feedback = self._thermal.compute_reactivity_feedback(
      self._temperature
    )


    return ReactorState(
      neutron_flux=self._flux,
      precursor_concentration=self._precursor,
      fuel_temperature=self._temperature,
      external_reactivity=self._external_reactivity,
      total_reactivity=self._external_reactivity + temp_feedback
    )
  
  def _get_observation(self) -> np.ndarray:
    # Construct the normalized observation vector

    flux_error = (self._flux - self._target_flux) / max(self._target_flux, 1e-10)
    flux_rate = (self._flux - self._prev_flux) / (self._env_cfg.dt * max(self._target_flux,1e-10))

    temp_dev = (self._temperature - self._phys_cfg.nominal_temperature) / self._phys_cfg.nominal_temperature

    precursor_ratio = self._precursor / max(self._kinetics.equilibrium_precursor(self._target_flux), 1e-10)

    ext_rho_normalized = self._external_reactivity / max(self._kinetics.beta, 1e-10)


    return np.array(
      [flux_error, flux_rate, temp_dev, precursor_ratio, ext_rho_normalized],
      dtype=np.float32   
    )
  


  def _check_safety_violation(self) -> bool:
    # Check if current state violates the safety contraints
    if self._temperature > self._env_cfg.max_temperature:
      return True
    if self._temperature < self._env_cfg.min_temperature:
      return True
    flux_ratio = self._flux / max(self._env_cfg.nominal_flux, 1e-10)
    if flux_ratio > self._env_cfg.max_flux_ratio:
      return True 
    if self._flux < self._env_cfg.min_flux:
      return True 
    return False 
  

  def _sample_disturbance(self) -> float: 
    # Sample a random reactivity disturbance if enabled.
    if not self._env_cfg.enable_disturbances:
      return 0.0 
    if self._rng.random() < self._env_cfg.disturbance_probability:
      return self._rng.uniform(
        -self._env_cfg.disturbance_magnitude,
        self._env_cfg.disturbance_magnitude  
      )
    return 0.0 
  

  def _build_info(
      self,
      total_reactivity: float, 
      temp_feedback: float,
      is_violation: bool, 
      disturbance: float, 
  ) -> Dict[str, Any]:
    # Build the info dictionary for the step return 

    return {
      "flux": self._flux,
      "temperature": self._temperature,
      "precursor": self._precursor,
      "external_reactivity": self._external_reactivity,
      "total_reactivity": total_reactivity,
      "temp_feedback": temp_feedback,
      "safety_violation": is_violation,
      "step": self._step_count,
      "disturbance": disturbance
    }
    