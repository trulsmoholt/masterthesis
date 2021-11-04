from discretization.mesh import Mesh
from discretization.FVML import compute_matrix,compute_vector
import numpy as np
import math

mesh = Mesh(10,10,lambda p: np.array([p[0],0.5*p[0]+p[1]]),ghostboundary=True)
source = lambda x,y:math.sin(y)*math.cos(x)
boundary_condition = lambda x,y:0
tensor = np.eye(2)
permeability = np.ones((mesh.num_unknowns,mesh.num_unknowns))

A = np.zeros((mesh.num_unknowns,mesh.num_unknowns))
f = np.zeros(mesh.num_unknowns)

compute_matrix(mesh,A,tensor,permeability)
compute_vector(mesh,f,source,boundary_condition)

u = np.linalg.solve(A,f)
mesh.plot_vector(u)