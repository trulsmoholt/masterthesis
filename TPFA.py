import numpy as np


from scipy.sparse import csr_matrix,lil_matrix

from mesh import Mesh


def compute_matrix(mesh,K,matrix,k_global=None,flux_matrix=None):
    nodes = mesh.nodes
    cell_centers = mesh.cell_centers
    nx = nodes.shape[1]
    ny = nodes.shape[0]
    if k_global is None:
        k_global = np.ones((cell_centers.shape[0],cell_centers.shape[1]))

    meshToVec = mesh.meshToVec
    if flux_matrix is not None:
        flux_matrix_x = flux_matrix['x']
        flux_matrix_y = flux_matrix['y']

    def local_assembler(j,i,vec,matrix_handle,index):
        
        matrix_handle[index,meshToVec(j-1,i-1)] += vec[0]
        matrix_handle[index,meshToVec(j-1,i)] += vec[1]
        matrix_handle[index,meshToVec(j,i)] += vec[2]
        matrix_handle[index,meshToVec(j,i-1)] += vec[3]


    for i in range(1,nodes.shape[0]-1):
        for j in range(1,nodes.shape[1]-1):
            omega = np.zeros((4,4,2))
            interface = np.zeros((4,2))
            centers = np.zeros((4,2))
            n = np.zeros((4,2))
            k_loc = np.zeros((4))

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

            V = np.zeros((4,2,2))

            for jj in range(4):
                i_2 = interface[jj-1]
                i_1 = interface[jj]

                X = np.array([i_1-centers[jj],i_2-centers[jj]])
                V[jj,:,:] = np.linalg.inv(X)
                

            for ii in range(4):
                for jj in range(4):
                    for kk in range(2):
                        omega[ii,jj,kk] = -n[ii,:].T@K@V[jj,:,kk]*k_loc[jj]
            
            #print(omega)

            A = np.array([[omega[0,0,0]-omega[0,1,1],0        ,0                          ,         0      ],
                        [       0  ,omega[1,1,0]-omega[1,2,1]  ,         0   ,0                          ],
                        [0                        ,         0      ,omega[2,2,0]-omega[2,3,1]  ,      0   ],
                        [     0       ,0                          ,        0    ,+omega[3,3,0]-omega[3,0,1]  ]])
            


            B = np.array([[omega[0,0,0] ,-omega[0,1,1] ,0                          ,0                          ],
                        [0                         ,omega[1,1,0]  ,-omega[1,2,1] ,0                          ],
                        [0                         ,0                          ,omega[2,2,0] ,-omega[2,3,1] ],
                        [-omega[3,0,1],0                          ,0                          ,omega[3,3,0]  ]])



            C = np.array([[omega[0,0,0],0           ,0           ,0],
                        [0,omega[1,1,0],0           ,0           ],
                        [0           ,0,omega[2,2,0],0           ],
                        [0           ,0           ,0,omega[3,3,0]]])       


            D = np.array([[omega[0,0,0],0                        ,0                        ,0                        ],
                        [0                        ,omega[1,1,0],0                        ,0                        ],
                        [0                        ,0                        ,omega[2,2,0],0                        ],
                        [0                        ,0                         ,0                       ,omega[3,3,0]]])



            T = C@np.linalg.inv(A)@B-D

            assembler = lambda vec,matrix,cell_index: local_assembler(i,j,vec,matrix,cell_index)

            assembler(T[0,:]+T[3,:],matrix,meshToVec(i-1,j-1))

            assembler( T[1,:] + -T[0,:],matrix,meshToVec(i-1,j))

            assembler(-T[2,:]-T[1,:],matrix,meshToVec(i,j))

            assembler( -T[3,:]+T[2,:],matrix,meshToVec(i,j-1))

            if flux_matrix is not None:
                assembler(T[0,:],flux_matrix_x,meshToVec(i-1,j-1))
                assembler(T[2,:],flux_matrix_x,meshToVec(i,j-1))
                assembler(T[3,:],flux_matrix_y,meshToVec(i-1,j-1))
                assembler(T[1,:],flux_matrix_y,meshToVec(i-1,j))


    for i in range(cell_centers.shape[0]):
        for j in range(cell_centers.shape[1]):
            if (i==0) or (i==ny-2) or (j==0) or (j==nx-2):
                matrix[meshToVec(i,j),:] = 0

                matrix[meshToVec(i,j),meshToVec(i,j)] = 1
    if flux_matrix is not None:
        return (matrix, flux_matrix_x, flux_matrix_y)
    return matrix



def compute_vector(mesh,f,boundary):
    nodes = mesh.nodes
    cell_centers = mesh.cell_centers
    num_unknowns = cell_centers.shape[1]*cell_centers.shape[0]
    nx = nodes.shape[1]
    ny = nodes.shape[0]
    meshToVec = mesh.meshToVec
    vecToMesh = mesh.vecToMesh
    vector = np.zeros(num_unknowns)
    h_y = nodes[1,0,1]-nodes[0,0,1]
    for i in range(cell_centers.shape[0]):
        for j in range(cell_centers.shape[1]):

            if (i==0) or (i==ny-2) or (j==0) or (j==nx-2):
                vector[meshToVec(i,j)]= boundary(cell_centers[i,j,0],cell_centers[i,j,1])
                continue
            vector[meshToVec(i,j)] += mesh.volumes[i,j]*f(cell_centers[i,j,0],cell_centers[i,j,1])
    return vector

if __name__=='__main__':
    import sympy as sym
    from differentiation import gradient,divergence
    from FVMO import compute_matrix as O
    import math
    x = sym.Symbol('x')
    y = sym.Symbol('y')
    K = np.array([[1,0],[0,1]])
    u_fabric = sym.cos(y*math.pi)*sym.cosh(x*math.pi)
    source = -divergence(gradient(u_fabric,[x,y]),[x,y],permability_tensor=K)
    source = sym.lambdify([x,y],source)
    u_lam = sym.lambdify([x,y],u_fabric)

    mesh = Mesh(6,6,lambda p: np.array([p[0]+0.5*p[1] ,p[1]]))
    mesh.plot()
    AT = np.zeros((mesh.num_unknowns,mesh.num_unknowns))
    flux_matrixT = {'x': np.zeros((mesh.num_unknowns,mesh.num_unknowns)),'y':np.zeros((mesh.num_unknowns,mesh.num_unknowns))}
    AT,fxT,fyT = compute_matrix(mesh,np.array([[1,0],[0,1]]),AT)
    AO = np.zeros((mesh.num_unknowns,mesh.num_unknowns))
    flux_matrixO = {'x': np.zeros((mesh.num_unknowns,mesh.num_unknowns)),'y':np.zeros((mesh.num_unknowns,mesh.num_unknowns))}
    AO,fxO,fyO = O(mesh,np.array([[1,0],[0,1]]),AO,flux_matrixO)
    diff = fyO-fyT
    sumA = AO+AT
    f = compute_vector(mesh,source,u_lam)
    ut = np.linalg.solve(AT,f)
    mesh.plot_vector(ut,'TPFA')
    f = compute_vector(mesh,source,u_lam)
    uo = np.linalg.solve(AO,f)
    mesh.plot_vector(uo,'MPFA-O')
    mesh.plot_vector(ut-uo,'difference')
    mesh.plot_funtion(u_lam,'exact solution')



