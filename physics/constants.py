from dataclasses import dataclass

@dataclass
class ReactorState:
  neutron_flux: float
  precursor_concentration: float
  fuel_temperature: float 
  external_reactivity: float 
  total_reactivity: float 

  def validate(self) -> bool:
    if self.neutron_flux < 0:
      return False 
    if self.precursor_concentration < 0:
      return False 
    if self.fuel_temperature < 0:
      return False 
    return True 