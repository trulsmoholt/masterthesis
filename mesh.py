import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay,ConvexHull
import math
import random


class Mesh:
    def __init__(self,num_nodes_x,num_nodes_y,P,ghostboundary = False):
        self.num_nodes_x = num_nodes_x
        self.num_nodes_y = num_nodes_y
        bottom_left = (0,0)
        top_right = (1,1)
        if ghostboundary:
            h_x = 1/(num_nodes_x-1)
            h_y = 1/(num_nodes_y-1)
            nodes_x, nodes_y = np.meshgrid(np.linspace(bottom_left[0]-h_x,top_right[0]+h_x,num=num_nodes_x+2),np.linspace(bottom_left[1]-h_y,top_right[1]+h_y,num=num_nodes_y+2))
        else:
            nodes_x, nodes_y = np.meshgrid(np.linspace(bottom_left[0],top_right[0],num=num_nodes_x),np.linspace(bottom_left[1],top_right[1],num=num_nodes_y))
        nodes = np.stack([nodes_x,nodes_y],axis=2)

        self.nodes = self.__perturb(nodes,P)
        self.cell_centers = self.__compute_cell_centers(self.nodes)
        self.volumes = self.__compute_volumes(self.nodes)
        self.midpoints = self.__compute_interface_midpoints(self.nodes)
        self.normals = self.__compute_normals(self.nodes,self.midpoints)
        self.num_unknowns  = self.cell_centers.shape[0]*self.cell_centers.shape[1]
        #self.elements,self.boundary_elements = self.__compute_triangulation(self.cell_centers,delaunay= False)


    # def __perturb(self,nodes, P):
    #     for y,row in enumerate(nodes):
    #         transform = lambda x: P(x,y/(self.num_nodes_y-1))
    #         T = np.vectorize(transform)
    #         nodes[y,:,0] = T(row[:,0])
    #     return nodes
    def __perturb(self,nodes,P):
        for i in range(nodes.shape[0]):
            for j in range(nodes.shape[1]):
                nodes[i,j,:] = P(nodes[i,j,:])
        return nodes

    def __compute_cell_centers(self,nodes):
        num_nodes_y = nodes.shape[0]
        num_nodes_x = nodes.shape[1]
        cell_centers = np.zeros((num_nodes_y-1,num_nodes_x - 1,2))
        for i in range(num_nodes_y-1):
            for j in range(num_nodes_x-1):

                x = (nodes[i,j]+nodes[i+1,j]+nodes[i,j+1]+nodes[i+1,j+1])*0.25
                cell_centers[i,j] = np.array([x[0],x[1]])
        return cell_centers
    def __compute_volumes(self,nodes):
        num_nodes_y = nodes.shape[0]
        num_nodes_x = nodes.shape[1]
        V = np.zeros((num_nodes_y-1,num_nodes_x - 1))
        for i in range(num_nodes_y-1):
            for j in range(num_nodes_x-1):
                shoelace1 = nodes[i,j,0]*nodes[i,j+1,1]+nodes[i,j+1,0]*nodes[i+1,j+1,1]+nodes[i+1,j+1,0]*nodes[i+1,j,1]+nodes[i+1,j,0]*nodes[i,j,1]
                shoelace2 = nodes[i,j,1]*nodes[i,j+1,0]+nodes[i,j+1,1]*nodes[i+1,j+1,0]+nodes[i+1,j+1,1]*nodes[i+1,j,0]+nodes[i+1,j,1]*nodes[i,j,0]
                V[i,j] = 0.5*abs(shoelace1-shoelace2)
        return V

    def __compute_interface_midpoints(self,nodes):
        num_nodes_y = nodes.shape[0]
        num_nodes_x = nodes.shape[1]
        midpoints = np.zeros((num_nodes_y-1,num_nodes_x-1,2,2))
        for i in range(num_nodes_y-1):
            for j in range(num_nodes_x-1):
                midpoints[i,j,0,:] = 0.5*(nodes[i+1,j,:]+nodes[i,j,:])
                midpoints[i,j,1,:] = 0.5*(nodes[i,j+1,:]+nodes[i,j,:])
        return midpoints

    def __compute_normals(self,nodes,midpoints):
        num_nodes_y = nodes.shape[0]
        num_nodes_x = nodes.shape[1]
        normals = np.zeros((num_nodes_y-1,num_nodes_x-1,2,2))
        for i in range(num_nodes_y-1):
            for j in range(num_nodes_x-1):
                v = midpoints[i,j,0,:]-nodes[i,j,:]
                normals[i,j,0,:] = np.array([v[1],-v[0]])
                v = midpoints[i,j,1,:]-nodes[i,j,:]
                normals[i,j,1,:] = np.array([-v[1],v[0]])
        return normals

    def __compute_triangulation(self,cell_centers, delaunay = True):
        boundary = np.zeros((cell_centers.shape[0],cell_centers.shape[1]))
        boundary[0,:] = 1
        boundary[cell_centers.shape[0]-1,:] = 3
        boundary[:,0] = 4
        boundary[:,cell_centers.shape[1]-1] = 2



        boundary = np.ravel(boundary)
        points = np.reshape(cell_centers,(cell_centers.shape[0]*cell_centers.shape[1],2),order='C')
        elements = np.zeros(((cell_centers.shape[0]-1)*(cell_centers.shape[1]-1)*2,3),dtype=int)
        e = 0
        if not delaunay:
            elements = np.zeros(((cell_centers.shape[0]-1)*(cell_centers.shape[1]-1)*2,3))
            e = 0
            for i in filter(lambda x: boundary[x]!=2 and self.num_unknowns-x>cell_centers.shape[1],range(self.num_unknowns)):
                elements[e,:] = np.array([int(i),int(i+cell_centers.shape[1]+1),int(i+cell_centers.shape[1])],dtype=int)
                e = e + 1
                elements[e,:] = np.array([int(i),int(i+1),int(i+cell_centers.shape[1]+1)],dtype=int)
                e = e + 1
            boundary_elements = ['n','n','n','n']
            for point, boundary_point,index in zip(points,boundary,range(points.shape[0])):
                if boundary_point != 0:
                    boundary_loc = int(boundary_point-1)
                    if type(boundary_elements[boundary_loc])==str:
                        boundary_elements[boundary_loc] = int(index)
                    elif isinstance(boundary_elements[boundary_loc],int):
                        boundary_elements[boundary_loc] = np.array([[boundary_elements[boundary_loc],index]])
                    else:
                        boundary_elements[boundary_loc] = np.concatenate((boundary_elements[boundary_loc],np.array([[boundary_elements[boundary_loc][boundary_elements[boundary_loc].shape[0]-1,1],index]])))
            bottom = boundary_elements[0]
            right = boundary_elements[1]
            top = boundary_elements[2]
            left = boundary_elements[3]
            bottom = np.concatenate((np.array([[left[0,0],bottom[0,0]]]),bottom,np.array([[bottom[bottom.shape[0]-1,1],right[0,0]]])))
            top= np.concatenate((np.array([[left[left.shape[0]-1,1],top[0,0]]]),top,np.array([[top[top.shape[0]-1,1],right[right.shape[0]-1,1]]])))
            boundary_elements = np.concatenate((bottom,right,top,left))
            return (elements,boundary_elements)
        T =  Delaunay(points)
        boundary_elements = ['n','n','n','n']
        for point, boundary_point,index in zip(points,boundary,range(points.shape[0])):
            if boundary_point != 0:
                boundary_loc = int(boundary_point-1)
                if type(boundary_elements[boundary_loc])==str:
                    boundary_elements[boundary_loc] = int(index)
                elif isinstance(boundary_elements[boundary_loc],int):
                    boundary_elements[boundary_loc] = np.array([[boundary_elements[boundary_loc],index]])
                else:
                    boundary_elements[boundary_loc] = np.concatenate((boundary_elements[boundary_loc],np.array([[boundary_elements[boundary_loc][boundary_elements[boundary_loc].shape[0]-1,1],index]])))
        bottom = boundary_elements[0]
        right = boundary_elements[1]
        top = boundary_elements[2]
        left = boundary_elements[3]
        bottom = np.concatenate((np.array([[left[0,0],bottom[0,0]]]),bottom,np.array([[bottom[bottom.shape[0]-1,1],right[0,0]]])))
        top= np.concatenate((np.array([[left[left.shape[0]-1,1],top[0,0]]]),top,np.array([[top[top.shape[0]-1,1],right[right.shape[0]-1,1]]])))
        boundary_elements = np.concatenate((bottom,right,top,left))
        boundary_set = set(boundary_elements.flatten())
        elements = T.simplices
        remove_list = []
        for i in range(elements.shape[0]):
            if elements[i,0] in boundary_set and elements[i,1] in boundary_set and elements[i,2] in boundary_set:
                remove_list.append(i)
        elements = np.delete(elements,remove_list,0)
        return (elements,boundary_elements)

    def max_h(self):
        m = 0
        for i in range(self.nodes.shape[0]-1):
            for j in range(self.nodes.shape[1]-1):
                up = np.linalg.norm(self.nodes[i+1,j,:]-self.nodes[i,j,:])
                right = np.linalg.norm(self.nodes[i,j+1,:]-self.nodes[i,j,:])
                diag1 = np.linalg.norm(self.nodes[i+1,j+1,:]-self.nodes[i,j,:])
                diag2 = np.linalg.norm(self.nodes[i+1,j,:]-self.nodes[i,j+1,:])
                m = max((m,up,right,diag1,diag2))
        return m
        

    def plot(self):
        #plt.scatter(self.cell_centers[:,:,0],self.cell_centers[:,:,1])
        plt.scatter(self.nodes[:,:,0], self.nodes[:,:,1])

        segs1 = np.stack((self.nodes[:,:,0],self.nodes[:,:,1]), axis=2)
        segs2 = segs1.transpose(1,0,2)
        plt.gca().add_collection(LineCollection(segs1))
        plt.gca().add_collection(LineCollection(segs2))
        # plt.quiver(*self.midpoints[1,1,0,:],self.normals[1,1,0,0],self.normals[1,1,0,1])
        # plt.quiver(*self.midpoints[1,1,1,:],self.normals[1,1,1,0],self.normals[1,1,1,1])
        #plt.savefig('perturbed_grid_aspect_0.2_mesh.pdf')
        points = np.reshape(self.cell_centers,(self.cell_centers.shape[0]*self.cell_centers.shape[1],2))


        #plt.triplot(points[:,0], points[:,1], self.elements,color = 'green',linestyle = 'dashed')
        # plt.savefig('figs/trapezoidal_mesh_1d5.pdf')

        plt.show()

    def meshToVec(self,j,i)->int:
        return i*self.cell_centers.shape[0] + j

    def vecToMesh(self,h)->(int,int):
        return (h % self.cell_centers.shape[0], math.floor(h/self.cell_centers.shape[0]))

    def plot_vector(self,vec,text = 'text'):
        vec_center = np.zeros((self.cell_centers.shape[0],self.cell_centers.shape[1]))
        num_unknowns = self.cell_centers.shape[1]*self.cell_centers.shape[0]
        for i in range(num_unknowns):
            vec_center[self.vecToMesh(i)] = vec[i]
        fig = plt.figure(figsize=plt.figaspect(0.5))
        plt.contourf(self.cell_centers[:,:,0],self.cell_centers[:,:,1],vec_center,20,)
        plt.colorbar()
        fig.suptitle(text)
        plt.savefig('Pressure.pdf')

        plt.show()
    def plot_interaction(self):
        #plot nodes
        plt.scatter(self.nodes[:,:,0], self.nodes[:,:,1])
        segs1 = np.stack((self.nodes[:,:,0],self.nodes[:,:,1]), axis=2)
        segs2 = segs1.transpose(1,0,2)
        plt.gca().add_collection(LineCollection(segs1))
        plt.gca().add_collection(LineCollection(segs2))

        centers = self.cell_centers
        points = np.zeros((2*centers.shape[0]-1,centers.shape[1],2))
        for i in range(centers.shape[1]-1):
            points[2*i,:,:] = centers[i,:,:]
            points[2*i+1,:,:] = self.midpoints[i,:,0,:]
        segs1 = np.stack((points[:,:,0],points[:,:,1]),axis=2)
        segs2 = segs1.transpose(1,0,2)
        plt.gca().add_collection(LineCollection(segs2,color='g',linestyle='dashed'))
        plt.show()



    def plot_funtion(self,fun,text = 'text'):
        vec_center = np.zeros((self.cell_centers.shape[0],self.cell_centers.shape[1]))
        num_unknowns = self.cell_centers.shape[1]*self.cell_centers.shape[0]
        for i in range(num_unknowns):
            xx,yy = self.cell_centers[self.vecToMesh(i)]
            vec_center[self.vecToMesh(i)] = fun(xx,yy)
        fig = plt.figure(figsize=plt.figaspect(0.5))
        plt.contourf(self.cell_centers[:,:,0],self.cell_centers[:,:,1],vec_center,20,)
        plt.colorbar()
        fig.suptitle(text)
        plt.show()
    
    def interpolate(self,fun):
        u = np.zeros(self.num_unknowns)
        for i in range(self.cell_centers.shape[0]):
            for j in range(self.cell_centers.shape[1]):
                x = self.cell_centers[i,j,0]
                y = self.cell_centers[i,j,1]
                u[self.meshToVec(i,j)] = fun(x,y)
        return u

    def compute_error(self,u,u_exact):
        u_exact_vec = u.copy()
        volumes = u.copy()
        for i in range(self.cell_centers.shape[0]):
            for j in range(self.cell_centers.shape[1]):
                u_exact_vec[self.meshToVec(i,j)] = u_exact(self.cell_centers[i,j,0],self.cell_centers[i,j,1])
                volumes[self.meshToVec(i,j)] = self.volumes[i,j]
        L2_error = math.sqrt(np.square(u-u_exact_vec).T@volumes/(np.ones(volumes.shape)@volumes))
        max_error = np.max(np.abs(u-u_exact_vec))
        return(L2_error,max_error)



    