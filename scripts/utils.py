import numpy as np
import astropy.units as u


def set_amplitude(config, energy):
    """
    Calculate the amplitude required for a driver to have the specified energy.
    Useful for comparing drivers with different widths without having them pump different amounts of energy into the simulation.

    Parameters
    ----------
    config : SACConfig object
        Configuration of simulation, including driver widths and density profile.

    energy : astropy.units.Quantity
        Amount of energy provided to the simulation by the driver.
    """

    
