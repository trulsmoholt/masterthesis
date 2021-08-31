import numpy as np
from numpy.lib.function_base import vectorize
from scipy.sparse import csr_matrix,lil_matrix
from mesh import Mesh
from numba import jit

def compute_matrix(mesh,K,matrix,k_global=None,flux_matrix = None):
    matrix.fill(0)
    nodes = mesh.nodes
    cell_centers = mesh.cell_centers
    if k_global is None:
        k_global = np.ones((cell_centers.shape[0],cell_centers.shape[1]))

    nx = nodes.shape[1]
    ny = nodes.shape[0]

    num_unknowns = cell_centers.shape[1]*cell_centers.shape[0]

    meshToVec = mesh.meshToVec
    if flux_matrix is not None:
        flux_matrix_x = flux_matrix['x']
        flux_matrix_y = flux_matrix['y']

    R = np.array([[0,1],[-1,0]])

    T = np.zeros((4,2,3))
    omega  = np.zeros((2,3,7))
    V = np.zeros((7,2))

    interface = np.zeros((4,2))
    centers = np.zeros((4,2))
    n = np.zeros((4,2))
    k_loc = np.zeros((4))

    def local_assembler(j,i,vec,start, matrix_handle,index):
        global_vec = np.zeros(num_unknowns)

        indexes = [meshToVec(j-1,i-1),meshToVec(j-1,i),meshToVec(j,i),meshToVec(j,i-1)]


        for ii,jj in zip(range(start,start+2),range(2)):

            matrix_handle[index,indexes[ii%4]] += vec[jj]

        matrix_handle[index,indexes[(start-1)%4]] += vec[2]
        return global_vec

    def compute_triangle_normals(start_index, interface, centers, node_midpoint,V):
        V[0,:] = R@(interface[(start_index-1)%4,:]-centers[start_index%4])
        V[1,:] = -R@(interface[start_index%4,:]-centers[start_index%4])
        V[2,:] = R@(interface[start_index%4,:]-centers[(start_index+1)%4])
        V[3,:] = -R@(nodes[i,j]-centers[(start_index+1)%4])
        V[4,:] = R@(nodes[i,j]-centers[(start_index-1)%4])
        V[5,:] = -R@(interface[(start_index-1)%4,:]-centers[(start_index-1)%4])
        V[6,:] = R@(nodes[i,j]-centers[start_index%4])


    # @jit(fastmath=True) #halfes the running time for matrix assembly, but no tensor permability
    def compute_omega(n,K,V,t,omega,center,k_loc_rel):
        for ii in range(2):
            for jj in range(3):
                for kk in range(7):
                    if ii == 0:
                        omega[ii,jj,kk] = n[center,:].T.dot(V[kk,:]*1/t[jj])*k_loc_rel[jj]
                    else:
                        omega[ii,jj,kk] = n[center-1,:].T.dot(V[kk,:]*1/t[jj])*k_loc_rel[jj]
        
    def compute_T(center,k_loc):
        compute_triangle_normals(center,interface,centers,nodes[i,j],V)
        t = np.array([V[0,:].T@R@V[1,:],V[2,:].T@R@V[3,:],V[4,:].T@R@V[5,:]])
        k_loc_rel = np.array([k_loc[center],k_loc[(center+1)%4],k_loc[(center-1)%4]])
        compute_omega(n,K,V,t,omega,center,k_loc_rel)

        xi_1 = (V[6,:].T@R@V[0,:])/(V[0,:].T@R@V[1,:])
        xi_2 = (V[6,:].T@R@V[1,:])/(V[0,:].T@R@V[1,:])

        C = np.array([[-omega[0,0,0],   -omega[0,0,1]],
                [-omega[1,0,0],   -omega[1,0,1]]])

        D = np.array([[omega[0,0,0]+omega[0,0,1],   0,  0],
                        [omega[1,0,0]+omega[1,0,1],   0,  0]])

        A = np.array([[omega[0,0,0]-omega[0,1,3]-omega[0,1,2]*xi_1,  omega[0,0,1]-omega[0,1,2]*xi_2],
                        [omega[1,0,0]-omega[1,2,5]*xi_1,  omega[1,0,1]-omega[1,2,4]-omega[1,2,5]*xi_2]])

        B = np.array([[omega[0,0,0]+omega[0,0,1]+omega[0,1,2]*(1-xi_1-xi_2),    -omega[0,1,2]-omega[0,1,3], 0],
                        [omega[1,0,0]+omega[1,0,1]+omega[1,2,5]*(1-xi_1-xi_2), 0,  -omega[1,2,4]-omega[1,2,5]]])

        T = C@np.linalg.inv(A)@B+D
        return T
        
    def choose_triangle(T,i):
        if abs(T[i,0,0])<abs(T[(i+1)%4,1,0]):
            return (T[i,0,:],i)
        else:
            return (T[(i+1)%4,1,:],(i+1)%4)


    for i in range(1,nodes.shape[0]-1):
        for j in range(1,nodes.shape[1]-1):

            #D
            v = nodes[i,j-1]-nodes[i,j]
            interface[3,:] = nodes[i,j] + 0.5*(v)
            n[3,:] = mesh.normals[i,j-1,1,:]
            centers[3,:] = cell_centers[i,j-1]

            #A
            v = nodes[i-1,j]-nodes[i,j]
            interface[0,:] = nodes[i,j] + 0.5*(v)
            n[0,:] = mesh.normals[i-1,j,0,:]
            centers[0,:] = cell_centers[i-1,j-1]

            #B
            v = nodes[i,j+1]-nodes[i,j]
            interface[1,:] = nodes[i,j] + 0.5*(v)
            n[1,:] = mesh.normals[i,j,1,:]
            centers[1,:] = cell_centers[i-1,j]

            #C
            v = nodes[i+1,j]-nodes[i,j]
            interface[2,:] = nodes[i,j] + 0.5*(v)
            n[2,:] = mesh.normals[i,j,0,:]
            centers[2,:] = cell_centers[i,j]

            k_loc[0] = k_global[i-1,j-1]

            k_loc[1] = k_global[i-1,j]

            k_loc[2] = k_global[i,j]

            k_loc[3] = k_global[i,j-1]

            for ii in range(4):
                T[ii,:,:] = compute_T(ii,k_loc)


            index = [meshToVec(i-1,j-1),meshToVec(i-1,j),meshToVec(i,j),meshToVec(i,j-1)]
            assembler = lambda vec,center,matrix,cell_index: local_assembler(i,j,vec,center,matrix, cell_index)

            m = {0:0,1:1,2:3,3:0}

            for jj in range(len(index)):
                sgn =( -1 if jj == 2 or jj == 3 else 1)
                t,choice = choose_triangle(T,jj)
                assembler(t*sgn,choice,matrix,index[jj])
                assembler(-t*sgn,choice,matrix,index[(jj+1)%4])
                
                # for computing flux over edges later
                if flux_matrix is not None:
                    if jj%2==0:
                        assembler(t,choice,flux_matrix_x,index[m[jj]])
                    else:
                        assembler(t,choice,flux_matrix_y,index[m[jj]])

                

    for i in range(cell_centers.shape[0]):
        for j in range(cell_centers.shape[1]):
            if (i==0) or (i==ny-2) or (j==0) or (j==nx-2):
                matrix[meshToVec(i,j),:] = 0
                matrix[meshToVec(i,j),meshToVec(i,j)] = 1




def compute_vector(mesh,f,boundary,vector):
    vector.fill(0)
    nodes = mesh.nodes
    cell_centers = mesh.cell_centers
    num_unknowns = cell_centers.shape[1]*cell_centers.shape[0]
    nx = nodes.shape[1]
    ny = nodes.shape[0]
    meshToVec = mesh.meshToVec
    for i in range(cell_centers.shape[0]):
        for j in range(cell_centers.shape[1]):
            if (i==0) or (i==ny-2) or (j==0) or (j==nx-2):
                vector[meshToVec(i,j)]= boundary(cell_centers[i,j,0],cell_centers[i,j,1])
                continue
            vector[meshToVec(i,j)] += mesh.volumes[i,j]*f(cell_centers[i,j,0],cell_centers[i,j,1])

if __name__=='__main__':
    import sympy as sym
    from differentiation import gradient,divergence
    import math
    x = sym.Symbol('x')
    y = sym.Symbol('y')
    K = np.array([[1,0],[0,1]])
    u_fabric = sym.cos(y*math.pi)*sym.cosh(x*math.pi)
    source = -divergence(gradient(u_fabric,[x,y]),[x,y],permability_tensor=K)
    source = sym.lambdify([x,y],source)
    u_lam = sym.lambdify([x,y],u_fabric)

    mesh = Mesh(20,20,lambda p: np.array([p[0] ,p[1]]))
    mesh.plot()
    A = np.zeros((mesh.num_unknowns,mesh.num_unknowns))
    flux_matrix = {'x': np.zeros((mesh.num_unknowns,mesh.num_unknowns)),'y':np.zeros((mesh.num_unknowns,mesh.num_unknowns))}
    permability = np.ones((mesh.cell_centers.shape[0],mesh.cell_centers.shape[1]))
    permability[2:4,16:19] = 0.1
    A,fx,fy = compute_matrix(mesh,np.array([[1,0],[0,1]]),A,permability,flux_matrix)
    f = compute_vector(mesh,source,u_lam)
    mesh.plot_vector(np.linalg.solve(A,f))
    mesh.plot_funtion(u_lam,'exact solution')

