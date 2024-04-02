import functions as fn
import datasets as ds
import connectomics as cs
from nilearn import image, maskers
from tqdm import tqdm
import sys
import numpy as np
from multiprocessing import Pool
import tempfile

def adjacent_voxels(x, y, z, shape):
    def _inbounds(coord, shape):
        if coord[0] > 0 and coord[0] < shape[0] and coord[1] > 0 and coord[1] < shape[1] and coord[2] > 0 and coord[2] < shape[2]:
            return True
        return False
    adj = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                adj_candidate = (x+i, y+j, z+k)
                if _inbounds(adj_candidate, shape):
                    adj.append(adj_candidate)
    return adj

def _generate_cones(args):
    idx = args[0]
    c = args[1]
    filename = args[2]
    brain_img = args[3]
    tc = args[4]
    memmap_shape = args[5]
    print(idx)
    coord_avgtcs = np.memmap(filename, mode="r+", shape=memmap_shape, dtype=np.float32)
    world_c = image.coord_transform(c[0], c[1], c[2], brain_img.affine)
    cone = masker.transform(fn.make_tms_cone(brain_img, world_c[0], world_c[1], world_c[2]))
    cone_avgtc = cs.extract_avg_signal(tc, cone)
    coord_avgtcs[idx,:] = cone_avgtc
    coord_avgtcs.flush()

if __name__=="__main__":
  
    print("loading files")
    mask = image.load_img(sys.argv[4]).get_fdata()
    brain_img = ds.get_img("MNI152_T1_2mm_brain_mask")
    brain = brain_img.get_fdata()
    
    masker = maskers.NiftiMasker(ds.get_img("MNI152_T1_2mm_brain_mask_dil")).fit()
    network = masker.transform("standard_data/anorexia_network.nii.gz")
    tc = masker.transform(sys.argv[1])
    print("files loaded")
    surface_dat = np.zeros(brain.shape)
    searchlight_result = np.zeros(brain.shape)
    surface_coords = []
    for c in np.ndindex(brain.shape):
        if brain[c] == 0 or mask[c] == 0:
            continue
        adj = adjacent_voxels(c[0], c[1], c[2], brain.shape)
        for a in adj:
            if brain[a] == 0:
                surface_dat[c] = 1
                surface_coords.append(c)
                break

    network_avgtc = cs.extract_avg_signal(tc, network)
    with tempfile.NamedTemporaryFile() as ntf:
        temp_name = ntf.name
        coord_avgtcs = np.memmap(temp_name, mode="w+", shape=(len(surface_coords),network_avgtc.shape[0]), dtype=np.float32)
        with Pool(16) as pool:
            args = [(i, surface_coords[i], temp_name, brain_img, tc, (len(surface_coords),network_avgtc.shape[0])) for i in range(0, len(surface_coords))]
            pool.map(_generate_cones, args)
        coord_avgtcs = coord_avgtcs.copy()
    searchlight_corrs = cs.pearson_corr(coord_avgtcs.T, network_avgtc[:,np.newaxis])
    searchlight_result = np.zeros(brain_img.shape)
    for idx, c in enumerate(surface_coords):
        searchlight_result[c] = searchlight_corrs[idx,0]
    max_coord = np.unravel_index(searchlight_result.argmax(), searchlight_result.shape) 
    max_mm_coord = image.coord_transform(max_coord[0], max_coord[1], max_coord[2], brain_img.affine)
    with open(sys.argv[3],"w+") as f:
        f.write(f"Peak vox coordinate: {max_coord} \n")
        f.write(f"Peak mm coordinate: {max_mm_coord} \n")
    image.new_img_like(brain_img, searchlight_result).to_filename(sys.argv[2])

    
