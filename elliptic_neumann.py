import numpy as np
import sympy as sym
import math

from FVML_neumann import compute_matrix , compute_vector
from differentiation import gradient, divergence
from mesh import Mesh
from operators import mass_matrix


mesh = Mesh(5,5,lambda p:np.array([p[0],p[1]]),ghostboundary=True)
x = sym.Symbol('x')
y = sym.Symbol('y')
u_exact = -0.1*sym.sin(math.pi*x)*sym.sin(math.pi*y)-1 #constructed solution
f = -divergence(gradient(u_exact,[x,y]),[x,y])
flux_density = gradient(u_exact,[x,y])


neumann_boundary = np.zeros(mesh.cell_centers.shape[1])
for j in range(1,mesh.cell_centers.shape[0]-1):
    n = mesh.normals[j,0,0,:]
    midpoint = mesh.midpoints[j,0,0,:]
    print(flux_density[0].subs([(x,midpoint[0]),(y,midpoint[1])]))
    normal_flux_density = flux_density[0].subs([(x,midpoint[0]),(y,midpoint[1])])*n[0] + flux_density[1].subs([(x,midpoint[0]),(y,midpoint[1])])*n[1]
    egde_length = np.linalg.norm(mesh.nodes[j+1,1]-mesh.nodes[j,1])
    neumann_boundary[j] = normal_flux_density*egde_length

f = sym.lambdify([x,y],f)
u_exact = sym.lambdify([x,y],u_exact)

u = np.zeros(mesh.num_unknowns)
b = np.zeros(mesh.num_unknowns)
A = np.zeros((mesh.num_unknowns,mesh.num_unknowns))
compute_vector(mesh,b,f,u_exact,neumann_boundary)
compute_matrix(mesh,A,np.eye(2))

for i in range(1,mesh.cell_centers.shape[0]-1):
    j = 0
    middle = mesh.meshToVec(i,j)
    bottom = mesh.meshToVec(i-1,j)
    top = mesh.meshToVec(i+1,j)
    bottom_egde = mesh.nodes[i,j]
    A[middle,top] = 



u = np.linalg.solve(A,b)
mesh.plot_vector(u,'computed solution')
mesh.plot_funtion(u_exact,'computed solution')
