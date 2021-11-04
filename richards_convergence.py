import numpy as np
import sympy as sym
import math

from discretization.TPFA import compute_matrix , compute_vector
from utils.differentiation import gradient, divergence
from discretization.mesh import Mesh
from discretization.operators import mass_matrix

def solve_richards(mesh: Mesh,tau=None):
    #construct solution
    K = np.array([[1,0],[0,1]])
    x = sym.Symbol('x')
    y = sym.Symbol('y')
    t = sym.Symbol('t')
    p = sym.Symbol('p')
    theta = sym.Piecewise(
        (p**(1/2)-p/2,p<=1),
        (1/2,p>0)
    ) #parametrized saturation
    kappa = 1 #parametrized permeability
    u_exact = 24*t**2*x**2*(1-x)*y**2*(1-y)+10**(-2)#constructed hydraulic head

    f = sym.diff(theta.subs(p,u_exact),t)-divergence(gradient(u_exact ,[x,y]),[x,y])#construct the force function given the exact solution

    #convert from sympy to regular python functions
    f = sym.lambdify([x,y,t],f)
    u_exact = sym.lambdify([x,y,t],u_exact)
    theta = sym.lambdify([p],theta)
    kappa = sym.lambdify([p],kappa)

    print('space discretization diameter, h: ',mesh.max_h(), 'timestep length: ',mesh.max_h()**2)
    #time discretization of 0-1
    timestep = mesh.max_h()#CFL condition
    time_partition = np.linspace(0,1,math.ceil(1/timestep)**2)
    tau = time_partition[1]-time_partition[0]

    #L-scheme parameters    
    L = 6
    TOL = 0.000000005

    u = mesh.interpolate(lambda x,y:u_exact(x,y,0))#initial condition

    #allocate storage
    u_l = u.copy() #L-scheme iterate
    u_t = u.copy() #timestep iterate
    F = u.copy() #source vector
    F.fill(0)
    A = np.zeros((mesh.num_unknowns,mesh.num_unknowns)) #stiffness matrix
    B = mass_matrix(mesh)
    compute_matrix(mesh, A, K)#compute stiffnes matrix
    #time iteration
    for t in time_partition[1:]:
        count=0
        #L-scheme iteration
        while True:
            count = count + 1
            #conductivity = kappa(np.reshape(u_l, (mesh.cell_centers.shape[0],mesh.cell_centers.shape[1]),order='F'))
            #A.fill(0)#empty the stiffness matrix
            lhs = L*B+tau*A
            F.fill(0)#empty load vector
            compute_vector(mesh,F,lambda x,y: f(x,y,t),lambda x,y:u_exact(x,y,t))#compute load vector
            rhs = L*B@u_l + B@theta(u_t) - B@theta(u_l) + tau*F
            u = np.linalg.solve(lhs,rhs)
            if np.linalg.norm(u-u_l)<=TOL+TOL*np.linalg.norm(u_l):
                break
            else:
                u_l = u
        print('L-scheme steps :',count)
        u_t = u
        u_l = u
        # mesh.plot_vector(u)
        # mesh.plot_funtion(lambda x,y:u_exact(x,y,t))
    mesh.plot_vector(u,'approximation')
    mesh.plot_funtion(lambda x,y:u_exact(x,y,t),'exact')
    
    return mesh.compute_error(u,lambda x,y : u_exact(x,y,1))

if __name__=="__main__":
    mesh = Mesh(10,10,lambda p: np.array([p[0],p[1]]))
    mesh.plot()
    print('L2 and max error:', solve_richards(mesh))