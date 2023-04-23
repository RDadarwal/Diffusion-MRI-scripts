#!/usr/bin/env python
# coding: utf-8

# # Diffusion DTI data analysis
# #### Last updated on: 23/04/2023
# #### Rakshit Dadarwal

# ### import libraries

import os
import nibabel as nib
import numpy as np
import pandas as pd

# DIPY
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from dipy.reconst.dti import fractional_anisotropy
from dipy.reconst.dti import color_fa


# ### Define path
# load MRI measurements scan list excel file
data_path = "/home/rdadarwal/"
excel_file = "ScanList.xlsx"

try:
    Siemens_meas = pd.read_excel(f"{data_path}{excel_file}", na_values="scalar")
except FileNotFoundError:
    # print("Oops!  File not found.  Check the file name again...")
    raise

# Siemens data folder type
folder_type = "human"  # select one out of: ['human', 'macaque', 'other']

# human measurement number (hum_0)
hum_nr = Siemens_meas["MRI Number"]

# bval file name
bval_file = "dwi.bval"
# bvec file name
bvec_file = "dwi.bvec"

# ### DTI model fitting

for hum in hum_nr.index:
    # raw data path
    raw_path = f"{data_path}/raw/{hum_nr[hum]}/"
    # load dwi data
    dwi_file = f"{raw_path}/eddy_corrected.nii.gz"

    # dwi files
    if not os.path.exists(dwi_file):
        print("Oops!  Preprocessed diffusion data not found.  Check the file again...")
        break
    img = nib.load(dwi_file)
    dwi = img.get_fdata()

    # bval files
    if not os.path.exists(f"{raw_path}/{bval_file}"):
        print("Oops!  bvalue file not found.  Check the file name again...")
        break
    # bvec files
    if not os.path.exists(f"{raw_path}/{bvec_file}"):
        print("Oops!  bvector file not found.  Check the file name again...")
        break

    # load bval and bvec files
    bvals, bvecs = read_bvals_bvecs(
        f"{raw_path}/{bval_file}", f"{raw_path}/{bvec_file}"
    )
    gtab = gradient_table(bvals, bvecs)

    # -----------------------------
    # Fit diffusion tensor model
    # -----------------------------
    ten_model = TensorModel(gtab)
    ten_fit = ten_model.fit(dwi)

    # --------------------------
    # Save DTI parametric maps
    # --------------------------
    if not os.path.exists(f"{data_path}/derivatives/{hum_nr[hum]}/DTI/"):
        os.makedirs(f"{data_path}/derivatives/{hum_nr[hum]}/DTI/")
    out_path = f"{data_path}/derivatives/{hum_nr[hum]}/DTI/"

    FA = ten_fit.fa
    AD = ten_fit.ad
    RD = ten_fit.rd
    MD = ten_fit.md
    evecs = ten_fit.evecs
    evals = ten_fit.evals

    for i in range(3):
        nib.save(
            nib.Nifti1Image((evecs[:, :, :, i, :]).astype(np.float32), img.affine),
            f"{out_path}/V{i+1}.nii.gz",
        )
        nib.save(
            nib.Nifti1Image((evals[:, :, :, i]).astype(np.float32), img.affine),
            f"{out_path}/L{i+1}.nii.gz",
        )

    nib.save(nib.Nifti1Image(FA, img.affine), f"{out_path}/FA.nii.gz")
    nib.save(nib.Nifti1Image(MD, img.affine), f"{out_path}/MD.nii.gz")
    nib.save(nib.Nifti1Image(RD, img.affine), f"{out_path}/RD.nii.gz")
    nib.save(nib.Nifti1Image(AD, img.affine), f"{out_path}/AD.nii.gz")

    # -----------------
    # Save FA RGB map
    # -----------------
    fa = fractional_anisotropy(ten_fit.evals)
    cfa = color_fa(fa, ten_fit.evecs)
    FA = np.clip(fa, 0, 1)
    RGB = color_fa(fa, ten_fit.evecs)

    nib.save(
        nib.Nifti1Image(np.array(255 * cfa, "uint8"), img.affine),
        f"{out_path}/FA_RGB.nii.gz",
    )
