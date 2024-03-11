import sys
from nilearn import image
import numpy as np

def center_of_gravity(arr):
    x = 0
    y = 0
    z = 0
    total = 0
    for c in np.ndindex(arr.shape):
        if arr[c] == 0:
            continue
        total += arr[c]
        x += c[0] * arr[c]
        y += c[1] * arr[c]
        z += c[2] * arr[c]
    return x / total, y / total, z / total

def top_cluster(conn_map_img, perc):
    sfg_mask = image.load_img("standard_data/sfg_mask.nii.gz").get_fdata()
    conn_map = conn_map_img.get_fdata()

    # Mask out non-sfg voxels
    masked = np.zeros(conn_map.shape)
    for i in np.ndindex(conn_map.shape):
        if sfg_mask[i] > 0:
            masked[i] = conn_map[i]

    # Threshold at 90%
    thresh = np.percentile(masked[masked > 0], perc)
    masked[masked < thresh] = 0
    cluster_img = image.new_img_like(conn_map_img, masked)

    # Get largest cluster
    largest_clust = image.largest_connected_component_img(cluster_img).get_fdata()

    clust_masked = np.zeros(conn_map.shape)
    for i in np.ndindex(conn_map.shape):
        if largest_clust[i] > 0:
            clust_masked[i] = masked[i]
    
    return clust_masked

if __name__=="__main__":
    depmap_img = image.load_img(sys.argv[1])
    clust_masked = top_cluster(depmap_img)
    target = center_of_gravity(clust_masked, 90)
    target_arr = np.zeros(depmap_img.shape)
    target_arr[round(target[0]), round(target[1]), round(target[2])] = 1
    image.new_img_like(depmap_img, target_arr).to_filename(sys.argv[2])
