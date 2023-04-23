#!/usr/bin/env python
# coding: utf-8

# # Diffusion NODDI data analysis
# #### Last updated on: 13/04/2023
# #### Rakshit Dadarwal

# ### import libraries

import os
import nibabel as nib
import pandas as pd

# DIPY
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table

# AMICO
import amico

amico.core.setup()


# ### Define path
# load MRI measurements scan list excel file
data_path = "/home/rdadarwal/FID_Studies/FID_RD_UMG/Neuromelanin/"
excel_file = "ScanList_test.xlsx"

try:
    Siemens_meas = pd.read_excel(f"{data_path}{excel_file}", na_values="scalar")
except FileNotFoundError:
    raise

# Siemens data folder type
folder_type = "human"  # select one out of: ['human', 'macaque', 'other']

# human measurement number (hum_0)
hum_nr = Siemens_meas["MRI Number"]

# bval file name
bval_file = "dwi.bval"
# bvec file name
bvec_file = "dwi.bvec"

# prefix
pref = "dwi"

# ### DTI model fitting

for hum in hum_nr.index[:1]:
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
    # Fit NODDI model
    # -----------------------------
    amico.util.fsl2scheme(f"{raw_path}/{bval_file}", f"{raw_path}/{bvec_file}")

    ae = amico.Evaluation(f"{data_path}/derivatives/", f"{hum_nr[hum]}")
    ae.load_data(
        dwi_filename=dwi_file,
        scheme_filename=f"{data_path}/raw/{pref}.scheme",
        b0_thr=10,
    )

    ae.set_model("NODDI")
    ae.generate_kernels()

    ae.load_kernels()

    ae.fit()
    ae.save_results()
