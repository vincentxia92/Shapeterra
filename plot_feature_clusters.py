# -*- coding: utf-8 -*-
import numpy as np
from mayavi import mlab
#用来 plot feature clusters

def plot_clusters(coord_array, tri_array, clusters, simil):
    """ plot_clusters plots clusters with same persistence similarity levels.
    :param coord_array: xyz coordinates per vertex in array of shape=(#points, 3)
    :param tri_array: vertex connection indices of the part in array of shape=(#triangles, 3)
    :param clusters: array of shape=(2*len(simil), #points) with cluster index/persistence values on even/odd rows
    :param simil: array with %similarity values, shape=(x,)
    :return:
    """

    # Create cluster plots for each similarity level
    for i in range(len(simil)):
        mlab.figure(figure= str(simil[i]) + "_Similarity_Clusters", bgcolor = (1,1,1), fgcolor = (0,0,0))

        # Create black wireframe mesh
        wire_mesh = mlab.triangular_mesh(coord_array[:, 0], coord_array[:, 1], coord_array[:, 2], tri_array, scalars=None,
                                         line_width=0.1, representation='wireframe', color=(0, 0, 0), opacity=0.1)

        # Create face coloring
        wire_mesh.mlab_source.dataset.cell_data
        wire_mesh.mlab_source.dataset.cell_data.scalars = cluster_cell_values(tri_array, clusters[2*i+1, :])
        wire_mesh.mlab_source.dataset.cell_data.scalars.name = 'Cell data'
        wire_mesh.mlab_source.update() 
        
        
        
        # Determine range colorbar
        unique_vals = np.unique(clusters[2*i+1, :])  # Sorted unique values of vertex_data
        face_val_min = unique_vals[1]*.995 # Adjust to show min val because 1st color lut is transparent and not shown
        face_val_max = unique_vals[-1]

        mesh = mlab.pipeline.set_active_attribute(wire_mesh, cell_scalars='Cell data')
        surf = mlab.pipeline.surface(mesh,vmin=face_val_min, vmax=face_val_max)
        
        # Retrieve the LUT colormap of the surf object. (256x4)
        # this is an array of (R, G, B, A) values (each in range 0-255)
        lut = surf.module_manager.scalar_lut_manager.lut.table.to_array()
    
        # Modify alpha channel lut for masking
        lut[:, -1] = np.ones((lut.shape[0]))*255*0.55
    
        # Add first row white color with full opacity to mask points with 0 value
        lut_mod = np.vstack((np.array([1, 1, 1, 0]), lut))
    
        # Now use this colormap and show colorbar
        surf.module_manager.scalar_lut_manager.lut.table = lut_mod

        # Add colorbar
        mlab.colorbar(surf, orientation='vertical')

        # Show plot
        mlab.show()

def cluster_cell_values(tri, cell_point_values):
    """ Function to select face values from lowest value corner points
    :param tri: vertex connection indices of the part triangular mesh
    :param cell_point_values: mesh triangle point values
    :return: mesh clus_face_val minimum values triangles
    """
    clus_face_val = np.zeros(tri.shape[0])
    for i in range(tri.shape[0]):
        clus_face_val[i] = np.amin(cell_point_values[tri[i, :]])  # amin
    return clus_face_val

if __name__ == "__main__":
    import os
    import sys
    
    # os.chdir(sys.path[0]) #change dir to main's path  
    # data = np.load('temp/coo.npz')
    # clusters = np.load('temp/clusters.npz')['clusters']
    # tri_array = data['tri_array'] ; coord_array = data['coord_array']
    # data.close()
    # simil = [0.7, 0.75, 0.8, 0.85, 0.9,.95] 
    # plot_clusters(coord_array, tri_array, clusters, simil)
