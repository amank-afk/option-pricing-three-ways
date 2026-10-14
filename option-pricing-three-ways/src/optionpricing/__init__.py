"""Pricing European and American options three ways.

* :mod:`optionpricing.black_scholes` - closed-form Black-Scholes prices and Greeks.
* :mod:`optionpricing.finite_difference` - explicit and theta-scheme (Crank-Nicolson)
  solvers for the Black-Scholes PDE in the log price.
* :mod:`optionpricing.trinomial` - a recombining trinomial tree that is the explicit
  scheme in disguise.
"""

from . import black_scholes, finite_difference, trinomial

__all__ = ["black_scholes", "finite_difference", "trinomial"]
__version__ = "1.0.0"
