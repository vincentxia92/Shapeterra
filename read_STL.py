import numpy as np
from stl import mesh

def readfromSTL(file): #takes a .stl file
    meshes=mesh.Mesh.from_file(file)
    alldata=meshes.data
    vectors=alldata['vectors']
    coord_array=np.reshape(vectors,(-1,3))
    coord_array,tri_array=np.unique(coord_array,return_inverse=True, axis=0)
    tri_array.resize(tri_array.size//3,3)
    return coord_array,tri_array


def writetoSTL(file,save_name): #takes a .npz file, output a .stl file
    data=np.load(file)
    faces= data['tri_array'].astype(np.int32) ; vertices = data['coord_array']
    data.close()
    # Create the mesh
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = vertices[f[j],:]
    
    # Write the mesh to file "cube.stl"
    cube.save(save_name)
    print("sucessfully converted to %s" %save_name)


if __name__ == "__main__":
    #writetoSTL('pocket_0.npz','pocket_0.stl')
    #writetoSTL('step_0.npz','step_0.stl')
    #writetoSTL('pyramid_0.npz','pyramid_0.stl')
    #writetoSTL('protrusion_0.npz','protrusion_0.stl')
    #writetoSTL('boss_0.npz','boss_0.stl')
    #writetoSTL('blind_hole_0.npz','blind_hole_0.stl')
    #writetoSTL('through_hole_0.npz','through_hole_0.stl')
    #writetoSTL("protrusion_1000.npz",'protrusion_1000.stl')
    coord_array,tri_array=readfromSTL('Covid_19.stl')
    print(coord_array,tri_array)
    #writetoSTL("./new_isolated/customize/customize_0001.npz",'customize_1000.stl')
    
    
