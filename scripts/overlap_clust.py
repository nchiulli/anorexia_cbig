from nimlab import functions as fn
from nilearn import image
import sys
from glob import glob


files = glob(f"{sys.argv[1]}/*.nii.gz")
imgs = [image.load_img(i) for i in files]

overlap = fn.get_overlap_map(imgs, 0)
overlap[0].to_filename("test_overlap.nii.gz")