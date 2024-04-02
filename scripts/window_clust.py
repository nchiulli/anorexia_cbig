from cluster_cog import top_cluster
from nilearn import image
import sys
import connectomics as cs
import functions as fn
import randrange


tc = image.load_img(sys.argv[1])
n_frames = tc.shape[3]
clust_imgs = []
for i in range(0, 25):
    k = randrange(0, n_frames - 240)
    sliced_tc = image.index_img(tc, slice(k, k+240))
    res = cs.singlesubject_seed_conn("standard_data/anorexia_network.nii.gz", sliced_tc, transform="zscore")
    clust = top_cluster(res, 99)
    clust_imgs.append(image.new_img_like(tc, clust))

fn.get_overlap_map(clust_imgs, 0)[0].to_filename(sys.argv[2])
