import numpy as np
import sympy as sym
import math

from FVML import compute_matrix , compute_vector
from differentiation import gradient, divergence
from mesh import Mesh
from operators import mass_matrix

def solve_richards(mesh: Mesh,tau=None):
    #construct solution
    K = np.array([[1,0],[0,1]])
    x = sym.Symbol('x')
    y = sym.Symbol('y')
    t = sym.Symbol('t')
    p = sym.Symbol('p')
    theta = 1/(1-p) #parametrized saturation
    kappa = p**2 #parametrized permeability
    u_exact = -0.1*sym.sin(math.pi*x)*sym.sin(math.pi*y)*t**2-1 #constructed hydraulic head

    f = sym.diff(theta.subs(p,u_exact),t)-divergence(gradient(u_exact ,[x,y]),[x,y],kappa.subs(p,u_exact))#construct the force function given the exact solution

    #convert from sympy to regular python functions
    f = sym.lambdify([x,y,t],f)
    u_exact = sym.lambdify([x,y,t],u_exact)
    theta = sym.lambdify([p],theta)
    kappa = sym.lambdify([p],kappa)

    print(mesh.max_h())
    #time discretization of 0-1
    timestep = mesh.max_h()**2#CFL condition
    time_partition = np.linspace(0,1,math.ceil(1/timestep))
    tau = time_partition[1]-time_partition[0]

    #L-scheme parameters    
    L = 0.5
    TOL = 0.000000005

    u = mesh.interpolate(lambda x,y:u_exact(x,y,0))#initial condition

    #allocate storage
    u_l = u.copy() #L-scheme iterate
    u_t = u.copy() #timestep iterate
    F = u.copy() #source vector
    A = np.zeros((mesh.num_unknowns,mesh.num_unknowns)) #stiffness matrix
    B = mass_matrix(mesh)

    #time iteration
    for t in time_partition[1:]:
        #L-scheme iteration
        while True:
            permeability = kappa(np.reshape(u_l, (mesh.cell_centers.shape[0],mesh.cell_centers.shape[1]),order='F'))
            A.fill(0)#empty the stiffness matrix
            compute_matrix(mesh, A, K,k_global=permeability)#compute stiffnes matrix
            lhs = L*B+tau*A
            F.fill(0)#empty force vector
            compute_vector(mesh,F,lambda x,y: f(x,y,t),lambda x,y:u_exact(x,y,t))#compute source vector
            rhs = L*B@u_l + B@theta(u_t) - B@theta(u_l) + tau*F
            u = np.linalg.solve(lhs,rhs)
            if np.linalg.norm(u-u_l)<=TOL+TOL*np.linalg.norm(u_l):
                break
            else:
                u_l = u
        u_t = u
        u_l = u
    return mesh.compute_error(u,lambda x,y : u_exact(x,y,1))

if __name__=="__main__":
    mesh = Mesh(10,10,lambda p: np.array([p[0],0.5*p[0] + p[1]]))
    mesh.plot()
    print(solve_richards(mesh))