# masterthesis
This repo contains discretization methods for elliptic PDEs in two spatial dimensions that handle general quadrilateral grids. It also contains implementations for solving Richards' equation. 
It requires numpy for assembling discretization matrices, for running the convergence tests one also needs scipy and sympy.

The code used for the figures in the masterthesis can be found in [elliptic](elliptic_convergence.py), [Richards' with constant hydraulic conductivity](richards_constant_hydraulic_conductivity.py) and [Richards' with non linear hydraulic conductivity](richards_non_linear.py). 
