from dataclasses import dataclass , field
from typing import List

@dataclass(frozen=True)
class PhysicsConstant:

  beta_total:float = 0.0064 
  lamda_precursor: float = 0.08 
  neutron_generation_time: float = 1e-4
  alpha_temp: float = -3.0e-5
  nominal_temperature: float = 573.15
  heat_capacity_coeff:float = 0.05
  heat_removal_coeff:float = 0.02
  coolant_inlet_temperature:float = 553.15


@dataclass(frozen=True)
class EnvironmentConfig:
  dt: float = 0.01 
  max_steps: int = 100
  nominal_flux: int = 1.0
  target_flux: int = 1.0
  max_temperature:float = 650.0
  min_temperature: float = 500.0
  max_flux_ratio:float = 1.5
  min_flux: float = 0.01
  reactivity_steps: float = 1e-4
  num_actions: int = 7
  enable_disturbances: bool = False 
  disturbance_magnitude: float = 5e-5
  disturbance_probability: float = 0.01
  seed: int = 67676767


@dataclass(frozen=True)
class RewardConfig:
  flux_error_weight: float = 10.0
  temp_deviation_weight: float = 1.0 
  control_effort_weight: float =  0.5
  on_target_bonus: float = 1.0
  on_target_tolerance: float = 0.02
  safety_violation_penalty: float = -100.0



@dataclass(frozen=True)
class DQNConfig:
  # Parameters for the DQN Agent
  learning_rate: float = 1e-3
  gamma: float = 0.99
  epsilon_start: float = 1.0
  epsilon_end: float = 0.01
  epsilon_decay_steps: int = 50000
  batch_size: int = 64
  replay_buffer_size: int = 100000
  target_update_interval: int = 1000
  hidden_layers: List[int] = field(default_factory=lambda: [128,128])
  gradient_clip_normalization: float = 1.0
  tau: float = 0.005 # Soft update coeff





@dataclass(frozen=True)
class TrainingConfig:
  num_episodes: int = 1000
  log_interval: int = 50
  save_interval: int = 200
  eval_interval: int = 100
  seed: int = 67676767




@dataclass(frozen=True)
class PIDConfig:
  kp: float = 5.0
  ki: float = 0.5
  kd: float = 0.1
  output_min: float = -5e-3
  output_max: float = 5e-3
  integral_limit: float = 1e-2


@dataclass(frozen=True)
class EvalConfig:
  num_episodes: int = 20
  setpoint_changes: List[float] = field(
    default_factory=lambda: [1.0,1.1,0.9,1.0]
  )
  setpoint_change_step: List[int] = field(
    default_factory=lambda: [0,1000,2500,4000]
  )