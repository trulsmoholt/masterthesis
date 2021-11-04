import numpy as np
import sympy as sym
import math

from discretization.FVMO import compute_matrix , compute_vector
from utils.differentiation import gradient, divergence
from discretization.mesh import Mesh
from discretization.operators import mass_matrix

mesh = Mesh(10,10,lambda p : np.array([p[0],p[0]*p[1]*0.5+p[1]]),ghostboundary=True)
mesh.plot()

K = np.array([[1,0],[0,1]])
x = sym.Symbol('x')
y = sym.Symbol('y')
t = sym.Symbol('t')
u_exact = -0.1*sym.sin(math.pi*x)*sym.sin(math.pi*y)*t**2-1
f = sym.diff(u_exact,t)-divergence(gradient(u_exact ,[x,y]),[x,y])

#convert from sympy to regular python functions
f = sym.lambdify([x,y,t],f)
u_exact = sym.lambdify([x,y,t],u_exact)


timestep=0.1

time_partition = np.linspace(0,1,math.ceil(1/timestep))
tau = time_partition[1]-time_partition[0]

u = mesh.interpolate(lambda x,y:u_exact(x,y,0))#initial condition
mesh.plot_vector(u)
#allocate storage
F = u.copy() #source vector
A = np.zeros((mesh.num_unknowns,mesh.num_unknowns)) #stiffness matrix
B = mass_matrix(mesh)

compute_matrix(mesh,A,K)

for t in time_partition[1:]:
    F.fill(0)#empty force vector
    compute_vector(mesh,F,lambda x,y: f(x,y,t),lambda x,y:u_exact(x,y,t))#compute source vector
    u = np.linalg.solve(B+tau*A,tau*F+B@u)
    # u = (B+tau*A)@u
    mesh.plot_vector(u,'u approximation')
    mesh.plot_funtion(lambda x,y:u_exact(x,y,t),'u exact')