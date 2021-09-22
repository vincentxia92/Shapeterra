import generate_hks
import plot_hks
import persistence
import plot_persistence
import cluster
import plot_clusters
import connection_matrix_point
import numpy as np
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#### Failed to build a .exe
##os.environ['ETS_TOOLKIT'] = 'qt4'
##import imp
##try:
##    imp.find_module('PySide') # test if PySide if available
##except ImportError:
##    os.environ['QT_API'] = 'pyqt' # signal to pyface that PyQt4 should be used
##from pyface.qt import QtGui, QtCore

import sys
print("This is the name of the script: ", sys.argv[0])
print ("Number of arguments: ", len(sys.argv))
print ("The arguments are: " , str(sys.argv))

import matplotlib.pyplot as plt
from stl import mesh
#from read_STEP import read_STEP,closed_shell
#import geometry_builder as gb
import save_adj
from read_STL import readfromSTL, writetoSTL
#import mesh_generator
#import mesh_from_step

print (sys.path)

import traceback

def main_onefile ( FILE_NAME): 
    try:
    
    
        # Read from STL
        coord_array,tri_array=readfromSTL(FILE_NAME) #'customize_0001.stl'#'%s.stl' %(sys.argv[1]) #./virus/coronavirus_anyaachan.stl works
    
        ## Write to STL
        #writetoSTL("./new_isolated/customize/customize_1000.npz",'./new_isolated/customize/customize_1000.stl')
      
        # Calculate HKS
        steps = 0.001
        iters = 1000
        HKS = generate_hks.generate_hks(coord_array, tri_array, steps, iters)
        
        #Save HKS for  
        n = [0, 10, 50, 299, 799]
        # np.savez_compressed('temp/HKS_%s'%FILE_NAME, HKS=HKS[:, n])
#        plot_hks.plot_hks(coord_array, tri_array, HKS, n, MIN = 0, MAX = 0.1)
        v = persistence.persistence(HKS) # PERSISTANCE VALUE
    
        #plot_mesh.plot_mesh(base_points, base_fv, "Persistence_value")
        plot_persistence.plot_persistence(coord_array, tri_array, v, 'value') # can specify MAX/MIN within the mayaki GUi=I
        
        # get the connection matrix of points, and save in file 'connection_matrix_points.npy'
        connection_matrix_point.connection_matrix(coord_array,tri_array)
        # Find clusters
        #simils = [0.7, 0.75, 0.8, 0.85, 0.9]  # Similarity percentages
        simils = [0.9]
        for simil in simils:
            adjm = np.array([])
            clusters = cluster.cluster(coord_array, tri_array, v, simil)  #v  # CLLUSTER ON HKS
            cluster_adjmap = cluster.get_attributed_cluster_adj_matrix(simil,clusters,tri_array)
            # Is this needed???
            # adjm = save_adj.get_feauture_cluster_adjm(model=None, cluster_adjmap,clusters) # only generate matrix for vertexes that is not on the boudary
            # simil -= 0.01 #fuzzy： 向下找  #simil += 0.01 # 
            print ('Adjacency Matrix has %d clusters in there.'%cluster_adjmap.shape[0])
            
            # save_adj.save_txt(FILE_NAME, adjm, "./temp/")     #fname,new_adjm,save_path
            
            plt.pcolor(cluster_adjmap, edgecolors='k', linewidths=1)
            #plt.xticks(range(60))
            #plt.yticks(range(60))
            plt.show()
            
            # Plot found clusters
            plot_clusters.plot_clusters(coord_array, tri_array, clusters, [simil]) # tales sim as a list
    except Exception as e:
        print(e)
        traceback.print_exc()

if __name__ == "__main__":
    
    main_onefile("Fuselage_tip.stl")
