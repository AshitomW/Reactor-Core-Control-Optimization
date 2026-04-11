# Reward function for reactor control environment.


from config import RewardConfig


class RewardCalculator:
  # COmpute shaped reward for reactor control actions.


  def __init__(self, reward_config: RewardConfig) -> None:
    self._cfg = reward_config


  def compute(
      self,
      flux: float,
      target_flux: float,
      fuel_temperature: float,
      nominal_temperature: float,
      reactivity_change: float,
      is_safety_violation: bool 
  ) -> float:
    """ Compute reward for the current transitions
    flux -> current neutron flux
    target_flux: desired neutron flux
    fuel_temperature: Current fuel temperature in Kelvin
    nominal_temperature: Nominal temperature in kelvin
    reactivity_change: Absolute reactivity changve applied this step.
    is_safety_violation: Whethere safety limits were breached?
""" 
    if is_safety_violation:
      return self._cfg.safety_violation_penalty
    

    # Flux tracking error (quadratic penatly)
    flux_error = (flux - target_flux) / max(target_flux, 1e-10)
    flux_penalty = -self._cfg.flux_error_weight * flux_error ** 2

    # Temperature deviation from the nominal (quadratic)
    temp_dev = (fuel_temperature - nominal_temperature) / nominal_temperature
    temp_penalty = -self._cfg.temp_deviation_weight * temp_dev ** 2

    # Control effort penalty
    effort_penalty = -self._cfg.control_effort_weight * abs(reactivity_change)

    # bonus for being close to thge target
    bonus = 0.0
    if abs(flux_error) < self._cfg.on_target_tolerance:
      bonus = self._cfg.on_target_bonus
    
    return flux_penalty + temp_penalty + effort_penalty + bonus 




