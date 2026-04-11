# Point kinetic model with one delayed neturon group

# Implements the coupled ordinary differential eqn:
# dn/dt = ((rho-beta)/Lambda) & n + lambda * C
# dC/dt = (beta/Lambda) * n - lambda * C
# n = neutron flux
# C = delayed neutron precursor concentration
# rho = total Reactivity
# beta = delayed neutron fraction
# Lambda = prompt neturon generation time
# lambda = precursor decay constant


import math
from typing import Tuple
from config import PhysicsConstant


class PointKineticsModel:
  # Solves point kinetics equn with one delayed neutron grou
  # with a semi implicit (exponential) integration method for numeric stability
  def __init__(self, constants: PhysicsConstant) -> None:
    self._beta = constants.beta_total
    self._lambda = constants.lamda_precursor
    self._gen_time = constants.neutron_generation_time

  @property
  def beta(self) -> float:
    return self._beta
  
  @property
  def lambda_precursor(self) -> float:
    return self._lambda
  
  @property
  def generation_time(self) -> float:
    return self._gen_time
  

  def compute_derivatives(
      self,
      neutron_flux: float,
      precursor_concentration: float, 
      total_reactivity: float 
  ) -> Tuple[float, float]:
    rho_minus_beta = total_reactivity - self._beta
    dn_dt = (rho_minus_beta / self._gen_time) * neutron_flux + self._lambda * precursor_concentration

    dc_dt = (self._beta / self._gen_time) * neutron_flux - self._lambda * precursor_concentration


    return dn_dt , dc_dt 
  

  def step(
      self,
      neutron_flux: float,
      precursor_concentration: float,
      total_reactivity: float,
      dt: float,
  ) -> Tuple[float, float]:
    '''
    Neutron flux and precursor concentration by one time step.

    Using a semi implicit method, the precursor eqn is solved analytically (exponential decay + source) and the flux equation will use implicit euler like treatment for the prompt term to maintain numerical stability even for small generation time.
    
    '''

    if dt<= 0:
      raise ValueError(f"Time step must be positive, got {dt}")
    

    # Precursor update, analytical solution of DC/dt = source - lamda * C


    decay_factor = math.exp(-self._lambda * dt)
    source_Rate = (self._beta / self._gen_time) * neutron_flux
    new_precursor = (precursor_concentration * decay_factor) + (source_Rate / self._lambda) * (1.0 - decay_factor)



    # Updating neutron flux, semi implicit for stability
    # dn/dt = [(rho-beta)/Lambda] * n  + lamda * C
    # implicit euler for prompt (potentially large) term:
    # n_new = n_old + dt * [(rho-beta)/Lambda * n_new + lamda * C_avg]
    # => n_new * [1-dt * (rho-beta)/Lambda] = n_old + dt * lambda * C_avg

    rho_minus_beta = total_reactivity - self._beta
    prompt_coeff = rho_minus_beta / self._gen_time
    precursor_avg = 0.5 * (precursor_concentration + new_precursor)
    denominator = 1.0 - dt * prompt_coeff

    # DO NOT DIVIDE BY 0
    if abs(denominator) < 1e-15:
      # fall black to explicit Euler with sub stepping
      new_flux = self._explicit_substep(
        neutron_flux, precursor_concentration , total_reactivity, dt
      )
    else:
      new_flux = (
        neutron_flux + dt * self._lambda * precursor_avg 
      ) / denominator

    # Clamping to prevent negative, flux is not negative 
    new_flux = max(new_flux, 0.0)
    new_precursor = max(new_precursor, 0.0)

    return new_flux, new_precursor
    
  def _explicit_substep(
      self,
      neutron_flux: float,
      precusor_concentration: float,
      total_reactivity: float,
      dt: float,
      num_substep: int = 100,
  ) -> float:
    '''
      Fallback explicit Euler integration with small substep
      used when the semi implicit denominator is near 0
    '''
    sub_dt = dt / num_substep
    n = neutron_flux
    c = precusor_concentration
    for _ in range(num_substep):
      dn, dc = self.compute_derivatives(n, c, total_reactivity)
      n = n + sub_dt * dn 
      c = c + sub_dt * dc 
      n = max(n,0.0)
      c = max(c,0.0)
    return n  
  

  def equilibrium_precursor(self, neutron_flux: float) -> float:
    """
      Compute the equilibrium precursor concentration for a given flux.
       At equilibrium: dC/dt = 0 => C = (beta / (Lambda * lambda)) * n
    """


    return (self._beta / (self._gen_time * self._lambda)) * neutron_flux


