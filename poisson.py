from discretization.mesh import Mesh 
import numpy as np
from discretization.FVML import compute_matrix,compute_vector
import math

nx = ny = 6
perturbation = lambda p:np.array([p[0],0.5*p[0]+p[1]])
mesh = Mesh(nx,ny,perturbation,ghostboundary=True)

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