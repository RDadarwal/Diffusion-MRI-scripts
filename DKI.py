#!/usr/bin/env python
# coding: utf-8

# # Diffusion DKI data analysis
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
import dipy.reconst.dki as dki

# ### Define path
# load MRI measurements scan list excel file
data_path = "/home/rdadarwal/"
excel_file = "ScanList.xlsx"

try:
    Siemens_meas = pd.read_excel(f"{data_path}{excel_file}", na_values="scalar")
except FileNotFoundError:
    # print("Oops!  File not found.  Check the file name again...")
    raise

# Bruker data folder type
folder_type = "human"  # select one out of: ['human', 'macaque', 'other']

# human measurement number (hum_0)
hum_nr = Siemens_meas["MRI Number"]

# bval file name
bval_file = "dwi.bval"
# bvec file name
bvec_file = "dwi.bvec"

# ### DKI model fitting

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
    # Fit diffusion kurtosis model
    # -----------------------------
    dkimodel = dki.DiffusionKurtosisModel(gtab)
    dkifit = dkimodel.fit(dwi)

    # --------------------------
    # Save DKI parametric maps
    # --------------------------
    if not os.path.exists(f"{data_path}/derivatives/{hum_nr[hum]}/DKI/"):
        os.makedirs(f"{data_path}/derivatives/{hum_nr[hum]}/DKI/")
    out_path = f"{data_path}/derivatives/{hum_nr[hum]}/DKI/"

    FA = dkifit.fa
    MD = dkifit.md
    RD = dkifit.rd
    AD = dkifit.ad

    MK = dkifit.mk(0, 3)
    AK = dkifit.ak(0, 3)
    RK = dkifit.rk(0, 3)

    nib.save(
        nib.Nifti1Image(FA, img.affine),
        os.path.join(out_path, "dki_FA.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(MD, img.affine),
        os.path.join(out_path, "dki_MD.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(RD, img.affine),
        os.path.join(out_path, "dki_RD.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(AD, img.affine),
        os.path.join(out_path, "dki_AD.nii.gz"),
    )

    nib.save(nib.Nifti1Image(AK, img.affine), os.path.join(out_path, "AK.nii.gz"))
    nib.save(nib.Nifti1Image(RK, img.affine), os.path.join(out_path, "RK.nii.gz"))
    nib.save(nib.Nifti1Image(MK, img.affine), os.path.join(out_path, "MK.nii.gz"))
