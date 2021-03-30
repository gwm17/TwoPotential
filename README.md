# TwoPotential Nuclear Decay Width Calculation
This is a python repository for calculating decay widths using the Two Potential approach and/or the WKB approximation (S. A. Gurvitz, *Novel Approach to Tunneling Problems*, Phys. Rev. A, 1988). For details on the theory see the Gurvitz article or S. Aberg, *Spherical proton emitters*, Phys. Rev. C, 1997.

# Python dependancies
The code is written for `<python 3>` and should not be used with `<python 2>`. The libraries `<scipy>`, `<numpy>`, `<matplotlib>`, and `<mpmath>` are required. These can all be found using your favorite python package manager, either `<pip>` or `<conda>`. The shebangs all specify the default user environment, so if you want to use a specific virtual environment, be sure to update these.

# Details
## Two Potential Approach
The two potential approach involves dividing the decay problem into two parts. First is the quasibound state in trapped in the potential well, and second is the particle tunneling through the barrier. This algorithim works by combining a Woods-Saxon and Coloumb potential, along with the centrifugal barrier, to model the nuclear potential. Optical model parameters are required as input, however the depth of the Woods-Saxon potential is only used as an initial guess. The depth of the potential is modified using a shooting method for the quasibound state such that the wave function goes to 0 within the classically forbbiden region. Once the potential depth is deterimned and the quasibound wave function is normalized, the scattering wave function is calculated using the regular Coulomb wave function. The results from these two methods are then used to calculate the decay width and half life.

The shooting method is implemented using the secant method for finiding roots. Numerov's method is used to solve the Schrodinger equation, and interpolation on the calculated wave function is done using the `<scipy>` library. `<mpmath>` provides the methods for calculating the normalized Coulomb wave function. The quasibound wave function is normalized using Simpsons rule from `<scipy>`.

## WKB Approximation
The WKB Approximation, or semi-classical limit, involves integrating the momentum over the barrier, which is defined by the classical turning points. Calculating the decay width also involves determining a prefactor, which is defined in the Aberg article. Integration is done using quadrature in the `<scipy>` package. Turning points are calculated by Brent's method of finding roots, again using the `<scipy>` implementation.

# Running the code
Assuming the proper environment is set, for the Two Potential approach simply run:
`<./TwoPotential.py>`
from within the TwoPotential directory. For the WKB Approximation run:
`<./WKB.py>`

