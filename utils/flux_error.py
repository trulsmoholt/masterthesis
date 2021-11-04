import numpy as np
from discretization.mesh import Mesh
import sympy as sym
import math
x = sym.Symbol('x')
y = sym.Symbol('y')


def compute_flux_error(fx,fy,u_fabric,mesh: Mesh):
    u_x = sym.diff(u_fabric,x)
    u_x = sym.lambdify([x,y],u_x)
    u_y = sym.diff(u_fabric,y)
    u_y = sym.lambdify([x,y],u_y)

    normals = mesh.normals
    midpoints = mesh.midpoints
    meshToVec = mesh.meshToVec
    vecToMesh = mesh.vecToMesh
    volumes = mesh.volumes

    fx_exact = np.zeros((midpoints.shape[0]*(midpoints.shape[1]-1)))
    vec = fx

    for i in range(normals.shape[0]):
        for j in range(0,normals.shape[1]-1):
            if i==0 or i == normals.shape[0]-1:
                fx_exact[meshToVec(i,j)] = vec[meshToVec(i,j)]
            else:
                grad_u = np.array([[u_x(midpoints[i,j+1,0,0],midpoints[i,j+1,0,1])],[u_y(midpoints[i,j+1,0,0],midpoints[i,j+1,0,1])]])
                fx_exact[meshToVec(i,j)] = -normals[i,j+1,0,:].T@grad_u/np.linalg.norm(normals[i,j+1,0,:])
                fx[meshToVec(i,j)] = fx[meshToVec(i,j)]/np.linalg.norm(2*normals[i,j+1,0,:])
    error_x = vec[:midpoints.shape[0]*(midpoints.shape[1]-1)] - fx_exact

    fy_exact = np.zeros((midpoints.shape[1]*(midpoints.shape[0]-1)))
    vec = fy
    for i in range(normals.shape[0]-1):
        for j in range(normals.shape[1]-1):
            if j==0 or j == normals.shape[1]-1:
                fy_exact[meshToVec(i,j)] = vec[meshToVec(i,j)]
            else:
                grad_u = np.array([[u_x(midpoints[i+1,j,1,0],midpoints[i+1,j+1,1,1])],[u_y(midpoints[i+1,j,1,0],midpoints[i+1,j+1,1,1])]])
                fy_exact[meshToVec(i,j)] = -normals[i+1,j,1,:].T@grad_u/np.linalg.norm(normals[i+1,j,1,:])
                fy[meshToVec(i,j)] = fy[meshToVec(i,j)]/np.linalg.norm(2*normals[i+1,j,1,:])
    error_y = vec[:midpoints.shape[1]*(midpoints.shape[0]-1)] - fy_exact


    max_error_x = np.max(np.abs(error_x))
    error_x = np.square(error_x)
    volumes_x = np.zeros(error_x.shape)
    for i in range(error_x.shape[0]):
        a,b = vecToMesh(i)
        if a==0 or a == normals.shape[0]-1:
            volumes_x[i] = 0
        else:
            volumes_x[i] = (volumes[a,b]+volumes[a,b-1])

    max_error_y = np.max(np.abs(error_y))
    error_y = np.square(error_y)
    volumes_y = np.zeros(error_y.shape)
    for i in range(error_y.shape[0]):
        a,b = vecToMesh(i)
        if  b == normals.shape[1]-1 or a==normals.shape[0]-1:
            volumes_y[i] = 0
        else:
            volumes_y[i] = (volumes[a,b]+volumes[a-1,b])
    l2_error = math.sqrt((volumes_x.T@error_x+volumes_y.T@error_y)/(np.ones(volumes_x.shape).T@volumes_x+np.ones(volumes_y.shape).T@volumes_y))

    return l2_error,max(max_error_x,max_error_y)