import numpy as np
import sympy as sym
import math
import matplotlib.pyplot as plt

from discretization.FVML import compute_matrix , compute_vector
from utils.differentiation import gradient, divergence
from discretization.mesh import Mesh
from discretization.operators import mass_matrix
"""
The code used for table 5.4 and 5.5
"""
def solve_richards(mesh: Mesh,timestep=None):
    #Van Genuchten parameteres
    a_g = 0.1844
    n_g = 3
    k_abs = 0.03
    my = 1
    exp_1 = n_g/(n_g-1)
    exp_2 = (n_g-1)/n_g

    #construct solution
    K = np.array([[1,0],[0,1]])#permeability tensor
    x = sym.Symbol('x')
    y = sym.Symbol('y')
    t = sym.Symbol('t')
    p = sym.Symbol('p')
    s = sym.Symbol('s')
    theta = sym.Piecewise(
        ((1+(-a_g*p)**n_g)**(-exp_2),p<=0),
        (1,p>0)
    ) #parametrized saturation
    kappa = ((k_abs/my)*s**(-1/2))*(1-(1-s**exp_1)**exp_2)**2#parametrized hydraulic conductivity as a function of saturation
    u_exact = -3*t*x*(1-x)*y*(1-y)-1#constructed hydraulic head
    f = sym.diff(theta.subs(p,u_exact),t)-divergence(gradient(u_exact ,[x,y]),[x,y],kappa.subs(s,theta.subs(p,u_exact)))#construct the force function given the exact solutio

    #convert from sympy to regular python functions
    f = sym.lambdify([x,y,t],f)
    u_exact = sym.lambdify([x,y,t],u_exact)
    kappa = sym.lambdify([p],kappa.subs(s,theta))
    theta = sym.lambdify([p],theta)

    #time discretization of 0-1
    time_partition = np.linspace(0,1,math.ceil(1/timestep))
    tau = time_partition[1]-time_partition[0]

    #L-scheme parameters    
    L = 0.3
    TOL = 0.000000005

    u = mesh.interpolate(lambda x,y:u_exact(x,y,0))#initial condition

    #allocate storage
    u_l = u.copy() #L-scheme iterate
    u_t = u.copy() #timestep iterate
    F = u.copy() #source vector
    F.fill(0)
    A = np.zeros((mesh.num_unknowns,mesh.num_unknowns)) #stiffness matrix
    B = mass_matrix(mesh)
    #time iteration
    for t in time_partition[1:]:
        count=0
        F.fill(0)#empty load vector
        compute_vector(mesh,F,lambda x,y: f(x,y,t),lambda x,y:u_exact(x,y,t))#compute load vector
        #L-scheme iteration
        while True:
            count = count + 1
            conductivity = kappa(np.reshape(u_l, (mesh.cell_centers.shape[0],mesh.cell_centers.shape[1]),order='F'))
            A.fill(0)#empty the stiffness matrix
            compute_matrix(mesh, A, K,conductivity)#compute stiffnes matrix
            lhs = L*B+tau*A
            rhs = L*B@u_l + B@theta(u_t) - B@theta(u_l) + tau*F
            u = np.linalg.solve(lhs,rhs)
            if np.linalg.norm(u-u_l)<=TOL+TOL*np.linalg.norm(u_l):
                break
            else:
                u_l = u
        err = mesh.compute_error(u,lambda x,y : u_exact(x,y,t))
        print('error at time ',t," : ",err[0])
        print('L-scheme iterations: ',count)
        u_t = u
        u_l = u
    
    return (mesh.max_h(),timestep,mesh.compute_error(u,lambda x,y : u_exact(x,y,1))[0])

if __name__=="__main__":
    number_of_tests = 3

    result = []
    for i in range(1,number_of_tests+1):
        num_nodes = 2*2**i
        mesh = Mesh(1+num_nodes,1+num_nodes,lambda p: np.array([p[0]-0.5*p[1],p[1]]),ghostboundary=True)
        result.append(solve_richards(mesh,mesh.max_h()))
    output=''
    for i in range(number_of_tests):
        row = result[i]
        if i==0:
            output = f'{i+1}&${row[0]:.5f}$&${row[1]:.5f}$&${row[2]:.6f}$&-\\\ \n'
        else:
            old_row = result[i-1]
            improvement = old_row[2]/row[2]
            output += f'{i+1}&${row[0]:.5f}$&${row[1]:.5f}$&${row[2]:.6f}$&${improvement:.5f}$\\\ \n'
    print(output)