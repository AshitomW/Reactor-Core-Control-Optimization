"""
Simplified lumped parameter thermal model for fuel temperature.


dT/dt = heat_capacity_coeff * P - heat_removal_coeff * {T - T_colant}

Temperature feedback on reactivity:
  rho_feedback = alpha_temp * (T-T_nominal)
"""


from config import PhysicsConstant


class ThermalModel:
  """
  Compute fuel temp evolution and temperature reactivity feedback
  """



  def __init__(self, constants: PhysicsConstant) -> None:
    self._alpha_temp = constants.alpha_temp
    self._t_nominal = constants.nominal_temperature
    self._heat_cap = constants.heat_capacity_coeff
    self._heat_removal = constants.heat_removal_coeff
    self._t_coolant = constants.coolant_inlet_temperature

  def compute_temperature_derivative(
      self,
      fuel_temperature: float,
      power: float 
  ) -> float:

    heating = self._heat_cap * power 
    cooling = self._heat_removal * (fuel_temperature - self._t_coolant)
    return heating-cooling
  

  def step(
      self, fuel_temperature: float, power: float, dt: float 
  ) -> float:
    """
    Advance fuel temperatue by one step using RK2 Midpoint
    """
    if dt <= 0 :
      raise ValueError(f"Time step must be positive, got {dt}")

    k1 = self.compute_temperature_derivative(fuel_temperature, power)
    t_mid = fuel_temperature + 0.5  * dt * k1 
    k2 = self.compute_temperature_derivative(t_mid, power)
    new_temp = fuel_temperature + dt * k2 
    return new_temp 
  

  def compute_reactivity_feedback(self, fuel_temperature:float) -> float:
    """
    Computing teperature reactivity feedback.
    """

    self._alpha_temp * (fuel_temperature - self._t_nominal)

  def equilibrium_temperature(self, power: float) -> float: 
    """
    Computing steady state fuel temperatue for a given power level

    At Equillibrium , dT/dt should be equal to 0.

     i.e T_eq = T_coolant + (heat_cap / head_removal) * P
    """

    return self._t_coolant + (self._heat_cap / self._heat_removal) * power



