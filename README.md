# masterthesis
This repo contains discretization methods for the [elliptic PDE](https://en.wikipedia.org/wiki/Elliptic_partial_differential_equation)

<img src="https://latex.codecogs.com/svg.image?u&plus;\nabla&space;\cdot&space;\left(&space;-\pmb{K}&space;\nabla&space;u\right)=f" title="u+\nabla \cdot \left( -\pmb{K} \nabla u\right)=f" />,

in two spatial dimensions that handle general quadrilateral grids. In particular, the MPFA-O and MPFA-L methods are implemented, they yield locally mass conservative discretizations that are consistent for rough grids. The MPFA-L method has particulary good monotonicity properties, i.e., the [maximum principle](https://en.wikipedia.org/wiki/Maximum_principle) is respected for a wide range of grids. It's therefore considered as the state of the art method for porous media flow problems on quadrilatereal grids.

The code requires numpy for assembling discretization matrices, for running the convergence tests one also needs scipy and sympy.
# Quick tutorial #
The way to define a domain and discretize it into control volumes, is to discretize the unit square with rectangles, then perturb the grid points. This approach is very flexible and alows for complicated domains and control volume discretizations.

```python
from discretization.mesh import Mesh
import numpy as np

nx = ny = 6 #Number of grid points in x and y direction on the unit square
perturbation = lambda p:np.array([p[0],0.5*p[0]+p[1]]) #perturbation on every grid point p
mesh = Mesh(nx,ny,perturbation,ghostboundary=True) #Creates a mesh object with a strip of ghostcells for boundary handling
mesh.plot()
```
This would result in the paralellogram discretization, note that we have 8 grid points (in orange) in each direction, 2 more than 6, as we have a strip of ghost cells.

![Figure_2_small](https://user-images.githubusercontent.com/49365904/145256307-a9b73542-e4ff-4c44-b6ff-0c6f63c6d8c3.png)

For solving the [Poisson equation ](https://en.wikipedia.org/wiki/Poisson%27s_equation) on this, one would define the problem data with numpy arrays and python funtions, then pass it to the compute_matrix and compute_vector functions, together with the mesh object. In this example we have a homogeneous domain (permeability is a matrix of ones), and a isotropic medium (tensor is diagonal).
```python
from discretization.FVML import compute_matrix,compute_vector
import math

source = lambda x , y : math.sin(y)*math.cos(x)
boundary_condition = lambda x , y :0
tensor = np.eye(2)
permeability = np.ones(( mesh.num_unknowns,mesh.num_unknowns))

A = np.zeros((mesh.num_unknowns,mesh.num_unknowns))# stiffness matrix
f = np.zeros(mesh . num_unknowns)# load vector

compute_matrix(mesh,A,tensor,permeability)
compute_vector(mesh,f,source,boundary_condition)

u = np.linalg.solve(A,f)
mesh.plot_vector(u)
```
This would result in the solution

![Figure_2_solution](https://user-images.githubusercontent.com/49365904/145258136-fcb74827-fa27-41f0-96aa-4711d4ca38c4.png)


For more interesting equations, such as the time dependent, non-linear [Richards' equation](https://en.wikipedia.org/wiki/Richards_equation), see [this file](richards_non_linear.py).
