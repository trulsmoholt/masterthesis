import numpy as np
import sympy as sym
import math

from FVML import compute_matrix as stiffness_matrix, compute_vector
from differentiation import gradient, divergence
from mesh import Mesh
from operators import mass_matrix

def solve_richards(mesh: Mesh):
    #construct solution
    K = np.array([[1,0],[0,1]])
    x = sym.Symbol('x')
    y = sym.Symbol('y')
    t = sym.Symbol('t')
    p = sym.Symbol('p')
    theta = 1/(1-p) #parametrized saturation
    kappa = p**2 #parametrized permeability
    u_exact = -0.1*sym.sin(math.pi*x)*sym.sin(math.pi*y)*t**2-1 #constructed hydraulic head

    f = sym.diff(theta.subs(p,u_exact),t)-divergence(gradient(u_exact ,[x,y]),[x,y],K.subs(p,u_exact))#construct the force function given the exact solution

    #convert from sympy to regular python functions
    f = sym.lambdify([x,y,t],f)
    u_exact = sym.lambdify([x,y,t],u_exact)
    theta = sym.lambdify([p],theta)
    kappa = sym.lambdify([p],kappa)

    #time discretization of 0-1
    time_partition = np.linspace(0,1,math.ceil(1/timestep))
    tau = time_partition[1]-time_partition[0]

    #L-scheme parameters    
    L = 0.5
    TOL = 0.000000005

    u = mesh.interpolate(lambda x,y:u_exact(x,y,0))#initial condition

    u_l = u.copy() #L-scheme iterate
    u_t = u.copy() #timestep iterate

    F = u.copy()
    A = np.zeros((mesh.num_unknowns,mesh.num_unknowns))
    B = mass_matrix(mesh)


    #time iteration
    for t in time_partition[1:]:
        while True:
            permeability = kappa(np.reshape(u_l, mesh.cell_centers.shape[0],mesh.cell_centers.shape[1],order='F'))
            lhs = L*B+tau*stiffness_matrix(mesh, K, A,k_global=permeability)
            F = compute_vector(mesh,lambda x,y: f(x,y,t),lambda x,y:u_exact(x,y,t),F)
            rhs = L*B@u_l - + B@theta(u_t) - B@theta(u_l) + tau*F
