import numpy as np
from read_from_off import read_off
import geometry_builder as gb

import plot_mesh
path = '/Volumes/My_Book_4TB/ModelNet10/bathtub/train/bathtub_0003.off'

pts_array,facets = read_off(path)

def check_norm(pts_array,facets):
    #some facets are actually on line
    p1 = pts_array[facets[:,0],:]
    p2 = pts_array[facets[:,1],:]
    p3 = pts_array[facets[:,2],:]
    v1 = p2 - p1 
    v2 = p3 - p2
    norm = np.cross(v1,v2)
    norm_norm = np.linalg.norm(norm,axis=1)
    
    return norm,norm_norm
    
norm,norm_norm = check_norm(pts_array,facets)

filtered = facets[np.where(norm_norm!=0)]

my_model = gb.solid_model(pts_array,filtered)
coord_array,tri_array = my_model.generate_mesh(mesh_length=1)

print(pts_array.shape,facets.shape)
#plot_mesh.plot_mesh(pts_array, facets)
plot_mesh.plot_mesh(coord_array, tri_array)