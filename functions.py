from __future__ import print_function

import os
import glob
import json
import nilearn
import numpy as np
import scipy as sp
import pandas as pd
import nibabel as nib
import multiprocessing

from numpy import ma
from scipy import stats
from random import random
from tqdm import tqdm, trange
import datasets as nimds
from scipy.spatial.distance import cdist
from statsmodels.regression.linear_model import OLS

from nilearn import plotting, image, regions, datasets, maskers, surface, masking

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

""" Miscellanious helpful functions.

Credits: Alex Cohen, Louis Soussand, Christopher Lin, and William Drew

"""

BRAIN_MASK = nimds.get_img("MNI152_T1_2mm_brain_mask_dil")


def sum_imgs(imgs):
    """Sum together a list of images.

    Parameters
    ----------
    imgs : list of Niimg-like objects
        Images to sum

    Returns
    -------
    Niimg-like object: Summed image

    """
    sum_dat = np.zeros(imgs[0].shape)
    for i in imgs:
        sum_dat += i.get_fdata()
    sum_img = image.new_img_like(imgs[0], sum_dat)
    return sum_img



def spatial_corr(img1, img2, mask=BRAIN_MASK):
    """Calculate spatial correlation between two images.

    Args:
        img1 (Niimg-like): First img
        img2 (Niimg-like): Second img
        mask (Niimg-like): binary mask img. Defaults to BRAIN_MASK (MNI152_T1_2mm_brain_mask_dil)

    Returns:
        float: Pearson's r correlation.
    """
    masker = maskers.NiftiMasker(mask, standardize=False).fit()
    vec1 = masker.transform(img1)[0, :]
    vec2 = masker.transform(img2)[0, :]
    return sp.stats.pearsonr(vec1, vec2)[0]


def make_sphere(ref_img, center, rad, val, mm_coord=False, strict_radius=False):
    """Create sphere centered on a coordinate.

    Args:
        ref_img (Niimg-like): Image to create sphere on
        center (3-tuple of int or float): Center of sphere
        rad (int or float): Radius of sphere
        val (float): Value to fill in sphere
        mm_coord (bool): Whether the center and radius are in millimeter (world) space.
            Defaults to False (voxel space, (0,0,0) is in a corner, not the center).
        strict_radius (bool): Whether to exclude voxels with center exactly `rad`
        distance away from the center. Defaults to False.

            For example,

            If strict_radius = False, then a 2mm radius will produce the following
            sphere (2mm template):

            [ ][X][ ]
            [X][X][X]
            [ ][X][ ]

            If strict_radius = True, then a 2mm radius will produce the following
            sphere:

            [ ][ ][ ]
            [ ][X][ ]
            [ ][ ][ ]

    """

    def _inbounds(coord, shape):
        if (
            coord[0] > 0
            and coord[0] < shape[0]
            and coord[1] > 0
            and coord[1] < shape[1]
            and coord[2] > 0
            and coord[2] < shape[2]
        ):
            return True
        return False

    def _strict_radius(strict_radius, dist, rad):
        if strict_radius:
            return dist < rad
        else:
            return dist <= rad

    dat = np.zeros(ref_img.shape)

    searchrad = rad
    if mm_coord:
        # Assume voxels are isotropic
        vox_size = ref_img.header.get_zooms()[0]
        searchrad = int(rad / vox_size) + 1

    inv_affine = np.linalg.inv(ref_img.affine)
    for c in np.ndindex((searchrad + 1) * 2, (searchrad + 1) * 2, (searchrad + 1) * 2):
        c_centered = (
            c[0] - (searchrad + 1),
            c[1] - (searchrad + 1),
            c[2] - (searchrad + 1),
        )
        img_origin = (0, 0, 0)
        if mm_coord:
            c_centered = image.coord_transform(
                c_centered[0], c_centered[1], c_centered[2], ref_img.affine
            )
            img_origin = image.coord_transform(
                img_origin[0], img_origin[1], img_origin[2], ref_img.affine
            )
        normalized_coord = [
            c_centered[0] - img_origin[0],
            c_centered[1] - img_origin[1],
            c_centered[2] - img_origin[2],
        ]
        dist = np.linalg.norm(normalized_coord)
        if _strict_radius(strict_radius, dist, rad):
            invol_coords = (
                center[0] + normalized_coord[0],
                center[1] + normalized_coord[1],
                center[2] + normalized_coord[2],
            )
            if mm_coord:
                invol_coords = image.coord_transform(
                    invol_coords[0], invol_coords[1], invol_coords[2], inv_affine
                )
            if _inbounds(invol_coords, ref_img.shape):
                dat[
                    int(round(invol_coords[0])),
                    int(round(invol_coords[1])),
                    int(round(invol_coords[2])),
                ] = val

    return image.new_img_like(ref_img, dat)


def make_tms_cone(ref_img, x, y, z, radii=[2, 4, 7, 9, 12], strict_radius=False):
    """Create a TMS cone at a coordinate

    Parameters
    ----------
    ref_img : Niimg-like
        Image to create sphere on
    x : int
        x coordinate in world-space mm
    y : int
        y coordinate in world-space mm
    z : int
        z coordinate in world-space mm
    radii : list of int
        List of radii (in mm) to use for spheres. Defaults to [2,4,7,9,12]
    strict_radius : bool
        If True, excludes voxels with center exactly `rad` distance away from
        the center. Defaults to False. See `make_sphere` for details.
    """
    blank_dat = np.zeros(ref_img.get_fdata().shape)
    blank_img = image.new_img_like(ref_img, blank_dat)
    spheres = []
    for r in radii:
        spheres.append(
            make_sphere(
                blank_img, (x, y, z), r, 1, mm_coord=True, strict_radius=strict_radius
            )
        )
    return sum_imgs(spheres)

