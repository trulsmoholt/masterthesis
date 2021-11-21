import numpy as np
import sympy as sym
import math
import random
from scipy.sparse import csr_matrix,lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from discretization.FVML import compute_matrix as compute_matrix_l
from discretization.FVMO import compute_matrix as compute_matrix_o
from discretization.TPFA import compute_matrix as compute_matrix_tpfa
from discretization.FVMO import compute_vector
from discretization.FEM import compute_matrix as compute_matrix_FEM
from discretization.FEM import compute_vector as compute_vector_FEM
from discretization.mesh import Mesh

from utils.flux_error import compute_flux_error
from utils.differentiation import divergence, gradient

def compute_error(mesh,u,u_fabric):
    cx = mesh.cell_centers.shape[1]
    cy = mesh.cell_centers.shape[0]
    u_fabric_vec = u.copy()
    volumes = u.copy()

    for i in range(cy):
        for j in range(cx):
            u_fabric_vec[mesh.meshToVec(i,j)] = u_fabric(mesh.cell_centers[i,j,0],mesh.cell_centers[i,j,1])
            volumes[mesh.meshToVec(i,j)] = mesh.volumes[i,j]
    #mesh.plot_vector(u-u_fabric_vec,'error')
    L2err = math.sqrt(np.square(u-u_fabric_vec).T@volumes/(np.ones(volumes.shape).T@volumes))
    maxerr = np.max(np.abs(u-u_fabric_vec))
    return (L2err,maxerr)

def random_perturbation(h,aspect):
    return lambda p: np.array([random.uniform(0,h)*random.choice([-1,1]) + p[0] - 0.5*p[1],(1/aspect)*random.uniform(0,h)*random.choice([-1,1]) + p[1]])

def chevron_perturbation(n):
    return lambda p: np.array([p[0],p[1]*0.5 + 0.5*(p[1]-1)*p[1]*0.3*math.sin(n*p[0])])

x = sym.Symbol('x')
y = sym.Symbol('y')
K = np.array([[1,0],[0,1]])
u_fabric = sym.cos(y*math.pi)*sym.cosh(x*math.pi)
source = -divergence(gradient(u_fabric,[x,y]),[x,y],permability_tensor=K)
source = sym.lambdify([x,y],source)
u_lam = sym.lambdify([x,y],u_fabric)

nx = 5
ny = 5
# T = lambda p: np.array([-0.5*p[1]+p[0],p[1]])
T = lambda p: np.array([p[0]-0.5*p[1],p[1]])
mesh = Mesh(20,40,chevron_perturbation(40),ghostboundary=True)
mesh.plot()





start = 3
end = 7
aspect = 1
result_pressure = np.zeros((end-start,9))
result_flux = np.zeros((end-start,9))

for i in range(start,end):
    result_pressure[i-start,0] = i
    result_flux[i-start,0] = i
    # mesh = Mesh(2**i,aspect*2**i,random_perturbation(1/(4*2**i),aspect),ghostboundary=True)
    # mesh = Mesh(2**i,2*2**i,chevron_perturbation(2*2**i),ghostboundary=True)
    mesh = Mesh(2**i,2**i,T)
    num_unknowns = mesh.num_unknowns


    #FVM-L
    A = lil_matrix((mesh.num_unknowns,mesh.num_unknowns))
    F = np.zeros(num_unknowns)
    flux_matrix = {'x': lil_matrix((num_unknowns,num_unknowns)),'y':lil_matrix((num_unknowns,num_unknowns))}
    compute_matrix_l(mesh,A,K,k_global=None,flux_matrix = flux_matrix)
    A = csr_matrix(A,dtype=float)
    compute_vector(mesh,F,source,u_lam)
    u = spsolve(A,F)
    fx = csr_matrix(flux_matrix['x'],dtype=float)
    fy = csr_matrix(flux_matrix['y'],dtype=float)
    l2err_flux,maxerr_flux = compute_flux_error(fx.dot(u),fy.dot(u),u_fabric,mesh)
    l2err,maxerr = compute_error(mesh,u,u_lam)
    result_pressure[i-start,1] = math.log(l2err,2)
    result_pressure[i-start,2] = math.log(maxerr,2)
    result_flux[i-start,1] = math.log(l2err_flux,2)
    result_flux[i-start,2] = math.log(maxerr_flux,2)


    #FEM
    A = lil_matrix((mesh.num_unknowns,mesh.num_unknowns))
    F = np.zeros(num_unknowns)
    compute_matrix_FEM(mesh,A,K,k_global=None)
    A = csr_matrix(A,dtype=float)
    f = compute_vector_FEM(mesh,F,source,u_lam)
    u = spsolve(A,f)
    u = np.reshape(u,(mesh.cell_centers.shape[0],mesh.cell_centers.shape[1]))
    u = np.ravel(u,order='F')

    l2err_flux,maxerr_flux = compute_flux_error(fx.dot(u),fy.dot(u),u_fabric,mesh)
    l2err,maxerr = compute_error(mesh,u,u_lam)
    result_pressure[i-start,7] = math.log(l2err,2)
    result_pressure[i-start,8] = math.log(maxerr,2)
    result_flux[i-start,7] = math.log(l2err_flux,2)
    result_flux[i-start,8] = math.log(maxerr_flux,2)

    #FVM-O
    A = lil_matrix((mesh.num_unknowns,mesh.num_unknowns))
    F = np.zeros(num_unknowns)
    flux_matrix = {'x': lil_matrix((num_unknowns,num_unknowns)),'y':lil_matrix((num_unknowns,num_unknowns))}
    compute_matrix_o(mesh,A,K,k_global=None,flux_matrix = flux_matrix)
    A = csr_matrix(A,dtype=float)
    compute_vector(mesh,F,source,u_lam)
    u = spsolve(A,F)
    fx = csr_matrix(flux_matrix['x'],dtype=float)
    fy = csr_matrix(flux_matrix['y'],dtype=float)
    l2err_flux,maxerr_flux = compute_flux_error(fx.dot(u),fy.dot(u),u_fabric,mesh)
    l2err,maxerr = compute_error(mesh,u,u_lam)
    result_pressure[i-start,3] = math.log(l2err,2)
    result_pressure[i-start,4] = math.log(maxerr,2)
    result_flux[i-start,3] = math.log(l2err_flux,2)
    result_flux[i-start,4] = math.log(maxerr_flux,2)


    #TPFA
    A = lil_matrix((mesh.num_unknowns,mesh.num_unknowns))
    F = np.zeros(num_unknowns)
    flux_matrix = {'x': lil_matrix((num_unknowns,num_unknowns)),'y':lil_matrix((num_unknowns,num_unknowns))}
    compute_matrix_tpfa(mesh,A,K,k_global=None,flux_matrix = flux_matrix)
    A = csr_matrix(A,dtype=float)
    compute_vector(mesh,F,source,u_lam)
    u = spsolve(A,F)
    fx = csr_matrix(flux_matrix['x'],dtype=float)
    fy = csr_matrix(flux_matrix['y'],dtype=float)
    l2err_flux,maxerr_flux = compute_flux_error(fx.dot(u),fy.dot(u),u_fabric,mesh)
    l2err,maxerr = compute_error(mesh,u,u_lam)
    result_pressure[i-start,5] = math.log(l2err,2)
    result_pressure[i-start,6] = math.log(maxerr,2)
    result_flux[i-start,5] = math.log(l2err_flux,2)
    result_flux[i-start,6] = math.log(maxerr_flux,2)




print(result_pressure)
print(result_flux)

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
fig.suptitle('potential error')
p3, = ax1.plot(result_pressure[:,0],result_pressure[:,7],'-o',color='red')
p3, = ax1.plot(result_pressure[:,0],result_pressure[:,3],'-v',color= 'g')

p1, = ax1.plot(result_pressure[:,0],result_pressure[:,1],'--x',color='k')
p3, = ax1.plot(result_pressure[:,0],result_pressure[:,5],'-*',color='y')
ax1.set_title('$L_2$ error')
ax1.grid()
ax1.set(xlabel='$log_2 n$',ylabel='$log_2 e$')


p4,=ax2.plot(result_pressure[:,0],result_pressure[:,8],'-o',color='red')
p4.set_label('FEM')
p4,=ax2.plot(result_pressure[:,0],result_pressure[:,4],'-v',color='g')
p4.set_label('O-method')
p2, = ax2.plot(result_pressure[:,0],result_pressure[:,2],'--x',color='k')
p2.set_label('L-method')
p4,=ax2.plot(result_pressure[:,0],result_pressure[:,6],'-*',color='y')
p4.set_label('TPFA-method')
ax2.grid()
ax2.set_title('$max$ error')

ax2.set(xlabel='$log_2 n$',ylabel='$log_2 e$')
plt.legend(loc='lower center',bbox_to_anchor=(0.0, -0.3),ncol=4)
fig.subplots_adjust(bottom=0.20)
plt.savefig('figs/pressure_chevron_grid.pdf')

plt.show()

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
fig.suptitle('normal flow error')
p3, = ax1.plot(result_flux[:,0],result_flux[:,7],'-o',color='r')
p3, = ax1.plot(result_flux[:,0],result_flux[:,3],'-v',color='g')

p1, = ax1.plot(result_flux[:,0],result_flux[:,1],'--x',color='k')
p3, = ax1.plot(result_flux[:,0],result_flux[:,5],'-*',color='y')

ax1.set_title('$L_2$ error')
ax1.grid()
ax1.set(xlabel='$log_2 n$',ylabel='$log_2 e$')


p4,=ax2.plot(result_flux[:,0],result_flux[:,8],'-o',color='r')
p4.set_label('FEM')
p4,=ax2.plot(result_flux[:,0],result_flux[:,4],'-v',color='g')
p4.set_label('O-method')
p2, = ax2.plot(result_flux[:,0],result_flux[:,2],'--x',color='k')
p2.set_label('L-method')

p4,=ax2.plot(result_flux[:,0],result_flux[:,6],'-*',color='y')
p4.set_label('TPFA-method')

ax2.grid()
ax2.set_title('$max$ error')

ax2.set(xlabel='$log_2 n$',ylabel='$log_2 e$')
plt.legend(loc='lower center',bbox_to_anchor=(0.0, -0.3),ncol=4)
fig.subplots_adjust(bottom=0.20)
plt.savefig('figs/flow_chevron_grid.pdf')

plt.show()
