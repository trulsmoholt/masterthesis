import numpy as np
import math
import sympy as sym
def compute_matrix(mesh,K,matrix,k_global = None,flux_matrix=None):
    elements = mesh.elements.astype(int)
    coordinates = np.reshape(mesh.cell_centers,(mesh.num_unknowns,2),order='C')
    boundary_elements_dirichlet = mesh.boundary_elements
    if k_global is None:
        k_global = np.ones((mesh.cell_centers.shape[0],mesh.cell_centers.shape[1]))
    k_global = np.ravel(k_global)
    shape_grad = np.array([np.matrix([[1],[0]]),np.matrix([[0],[1]]),np.matrix([[-1],[-1]])])
    if flux_matrix is not None:
        flux_matrix_x = flux_matrix['x']
        flux_matrix_y = flux_matrix['y']
    def local_to_reference_map(ele_num):
        mat_coords = np.array([[coordinates[elements[ele_num,0]][0],coordinates[elements[ele_num,0]][1],1],[coordinates[elements[ele_num,1]][0],coordinates[elements[ele_num,1]][1],1],[coordinates[elements[ele_num,2]][0],coordinates[elements[ele_num,2]][1],1]])
        b1 = np.array([[1],[0],[0]])
        b2 = np.array([[0],[1],[0]])
        a1 = np.linalg.solve(mat_coords,b1)
        a2 = np.linalg.solve(mat_coords,b2)
        J = np.matrix([[a1[0][0],a1[1][0]],[a2[0][0],a2[1][0]]])
        c = np.matrix([[a1[2][0]],[a2[2][0]]])
        return [J,c]
    def reference_to_local_map(ele_num):
        mat_coords = np.array([[1,0,1],[0,1,1],[0,0,1]])
        b1 = np.array([[coordinates[elements[ele_num][0]][0]],[coordinates[elements[ele_num][1]][0]],[coordinates[elements[ele_num][2]][0]]])
        b2 = np.array([[coordinates[elements[ele_num][0]][1]],[coordinates[elements[ele_num][1]][1]],[coordinates[elements[ele_num][2]][1]]])
        a1 = np.linalg.solve(mat_coords,b1)
        a2 = np.linalg.solve(mat_coords,b2)
        J = np.matrix([[a1[0][0],a1[1][0]],[a2[0][0],a2[1][0]]])
        c = np.matrix([[a1[2][0]],[a2[2][0]]])
        return [J,c]

    # Matrix assembly
    for e in range(len(elements)):
        # extract element information
        [J,c] = local_to_reference_map(e)
        [M,d] = reference_to_local_map(e)
        transform = J.dot(J.transpose()) #J*J^t; derivative transformation
        jac = np.linalg.det(J) #Determinant of tranformation matrix = inverse of area of local elements
        #Local assembler
        for j in range(3):
            if flux_matrix is not None:
                flux = M.dot(shape_grad[j])
                # flux_matrix_x[e,elements[e,j]] += flux[0]
                # flux_matrix_y[e,elements[e,j]] += flux[1]
            for i in range(3):
                x_r = M@np.array([[1/3],[1/3]])+d
                K_int = k_global[elements[e][2]]*(1-x_r[0]-x_r[1])+k_global[elements[e][0]]*x_r[0]+k_global[elements[e][1]]*x_r[1]
                # K_int = (k_global[elements[e][2]]+k_global[elements[e][0]]+k_global[elements[e][1]])/3
                # K_int = 0.5*k_global[elements[e][j]]
                matrix[elements[e][i],elements[e][j]] += K_int*0.5*shape_grad[i].transpose().dot(transform.dot(shape_grad[j]))/jac
    for e in range(len(boundary_elements_dirichlet)):
        matrix[boundary_elements_dirichlet[e][0],:]=0
        matrix[boundary_elements_dirichlet[e][0],boundary_elements_dirichlet[e][0]]=1
        matrix[boundary_elements_dirichlet[e][1],:]=0
        matrix[boundary_elements_dirichlet[e][1],boundary_elements_dirichlet[e][1]]=1
    if flux_matrix is not None:
        return (matrix,flux_matrix_x,flux_matrix_y)
    return matrix

def compute_vector(mesh,f,boundary):
    x = sym.Symbol('x')
    y = sym.Symbol('y')
    #s = sym.Symbol('s')

    f_vect = np.zeros(mesh.cell_centers.shape[0]*mesh.cell_centers.shape[1])
    elements = mesh.elements.astype(int)
    coordinates = np.reshape(mesh.cell_centers,(mesh.num_unknowns,2),order='C')
    shape_func = np.array([x,y,1-y-x])
    #shape_func_1d = np.array([0.5-0.5*x,0.5+0.5*x])

    boundary_elements_dirichlet = mesh.boundary_elements

    def local_to_reference_map(ele_num):
        mat_coords = np.array([[coordinates[elements[ele_num][0]][0],coordinates[elements[ele_num][0]][1],1],[coordinates[elements[ele_num][1]][0],coordinates[elements[ele_num][1]][1],1],[coordinates[elements[ele_num][2]][0],coordinates[elements[ele_num][2]][1],1]])
        b1 = np.array([[1],[0],[0]])
        b2 = np.array([[0],[1],[0]])
        a1 = np.linalg.solve(mat_coords,b1)
        a2 = np.linalg.solve(mat_coords,b2)
        J = np.matrix([[a1[0][0],a1[1][0]],[a2[0][0],a2[1][0]]])
        c = np.matrix([[a1[2][0]],[a2[2][0]]])
        return [J,c]

    def reference_to_local_map(ele_num):
        mat_coords = np.array([[1,0,1],[0,1,1],[0,0,1]])
        b1 = np.array([[coordinates[elements[ele_num][0]][0]],[coordinates[elements[ele_num][1]][0]],[coordinates[elements[ele_num][2]][0]]])
        b2 = np.array([[coordinates[elements[ele_num][0]][1]],[coordinates[elements[ele_num][1]][1]],[coordinates[elements[ele_num][2]][1]]])
        a1 = np.linalg.solve(mat_coords,b1)
        a2 = np.linalg.solve(mat_coords,b2)
        J = np.matrix([[a1[0][0],a1[1][0]],[a2[0][0],a2[1][0]]])
        c = np.matrix([[a1[2][0]],[a2[2][0]]])
        return [J,c]

    def quad_2d_2nd_order_shape(ele_num, f, loc_node):
        [J,c] = reference_to_local_map(ele_num)
        x_1 = J.dot(np.array([[1/6],[1/6]]))+c
        x_2 = J.dot(np.array([[2/3],[1/6]]))+c
        x_3 = J.dot(np.array([[1/6],[2/3]]))+c
        return (1/6)*(f(x_1[0][0],x_1[1][0])*shape_func[loc_node].subs([(x,1/6),(y,1/6)])+f(x_2[0][0],x_2[1][0])*shape_func[loc_node].subs([(x,2/3),(y,1/6)])+f(x_3[0][0],x_3[1][0])*shape_func[loc_node].subs([(x,1/6),(y,2/3)]))
    # #Parametrizes straight boundary segment to [-1,1]
    # def param_1d_ele(ele_num):
    #     return np.array([[coordinates[boundary_elements_neumann[ele_num][0]][0]+(coordinates[boundary_elements_neumann[ele_num][1]][0]-coordinates[boundary_elements_neumann[ele_num][0]][0])*0.5*(s+1)],[coordinates[boundary_elements_neumann[ele_num][0]][1]+(coordinates[boundary_elements_neumann[ele_num][1]][1]-coordinates[boundary_elements_neumann[ele_num][0]][1])*0.5*(s+1)]])
        
    # #Calculates length of 1d interval
    # def param_1d_ele_derivative(ele_num):
    #     #return math.sqrt((coordinates[boundary_elements_neumann[ele_num][0]][0]-coordinates[boundary_elements_neumann[ele_num][0]][1])^2+(coordinates[boundary_elements_neumann[ele_num][1]][0]-coordinates[boundary_elements_neumann[ele_num][1]][1])^2)
    #     return 0.5*math.sqrt((coordinates[boundary_elements_neumann[ele_num][0]][0]-coordinates[boundary_elements_neumann[ele_num][1]][0])**2+(coordinates[boundary_elements_neumann[ele_num][0]][1]-coordinates[boundary_elements_neumann[ele_num][1]][1])**2)
    # # Second order quadrature on boundary line integral
    # def quad_2nd_ord_line(f,ele_num,loc_node):
    #     r = param_1d_ele(ele_num)
    #     dr = param_1d_ele_derivative(ele_num)
    #     x_1 = r[0][0].subs(s,-1/math.sqrt(3))
    #     x_2 = r[0][0].subs(s,1/math.sqrt(3))
    #     y_1 = r[1][0].subs(s,-1/math.sqrt(3))
    #     y_2 = r[1][0].subs(s,1/math.sqrt(3))
    #     return (f(x_1, y_1)*shape_func_1d[loc_node].subs(x,-1/math.sqrt(3))+f(x_2,y_2)*shape_func_1d[loc_node].subs(x,1/math.sqrt(3)))*dr
    for e in range(len(elements)):
        # extract element information
        [J,c] = local_to_reference_map(e)
        jac = np.linalg.det(J) #Determinant of tranformation matrix = inverse of area of local elements
        #Local assembler
        for j in range(3):
            f_vect[elements[e][j]] = float(f_vect[elements[e][j]]) + quad_2d_2nd_order_shape(e,f,j)/jac  
    for e in range(len(boundary_elements_dirichlet)):
        f_vect[boundary_elements_dirichlet[e][0]]=boundary(coordinates[boundary_elements_dirichlet[e][0]][0], coordinates[boundary_elements_dirichlet[e][0]][1])
        f_vect[boundary_elements_dirichlet[e][1]]=boundary(coordinates[boundary_elements_dirichlet[e][1]][0], coordinates[boundary_elements_dirichlet[e][1]][1])
    return f_vect

    def compute_error(mesh,u,u_fabric):
        elements = mesh.elements

if __name__=='__main__':
    import sympy as sym
    from differentiation import gradient,divergence
    import math
    from mesh import Mesh
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
    print(permability)
    A,fx,fy = compute_matrix(mesh,np.array([[1,0],[0,1]]),A,permability,flux_matrix)
    f = compute_vector(mesh,source,u_lam)
    u = np.linalg.solve(A,f)
    u = np.reshape(u,(mesh.cell_centers.shape[0],mesh.cell_centers.shape[1]))
    u = np.ravel(u,order='F')
    mesh.plot_vector(u)
    mesh.plot_funtion(u_lam,'exact solution')