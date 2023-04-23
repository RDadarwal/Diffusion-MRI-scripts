#!/usr/bin/env python
# coding: utf-8

# # Diffusion MRI data preprocessing
# #### Last updated on: 23/04/2023
# #### Rakshit Dadarwal

# ### import libraries

import os
import glob
import shutil
import nibabel as nib
import numpy as np
import pandas as pd

# Nipype wrappers
from nipype.interfaces.dcm2nii import Dcm2niix
from nipype.interfaces import fsl

# DIPY
from dipy.denoise.patch2self import patch2self
from dipy.segment.mask import median_otsu

# ### Define path
# load MRI measurements scan list excel file
data_path = "/home/rdadarwal/"
excel_file = "ScanList.xlsx"
dicom_prefix = "H"  # prefix for dicom files ["H" - Human; "M" - Macaque]

try:
    Siemens_meas = pd.read_excel(f"{data_path}{excel_file}", na_values="scalar")
except FileNotFoundError:
    # print("Oops!  File not found.  Check the file name again...")
    raise

# Siemens data folder type
folder_type = "human"  # select one out of: ['human', 'macaque', 'other']

# Diffusion measurement number
meas_nr = Siemens_meas["dMRI_2iso"]
# Diffusion measurement with reverse phase encoding (e.g. PA) number
inv_meas_nr = Siemens_meas["dMRI_2iso_inv"]
# human measurement number (hum_0)
hum_nr = Siemens_meas["MRI Number"]

# number of b0 images
num_b0 = 1

# bval file name
bval_file = "dwi.bval"
# bvec file name
bvec_file = "dwi.bvec"

# ### create nifti
for hum in meas_nr.index:
    if (
        os.path.exists(f"/mnt/scanner/siemens/{folder_type}/{hum_nr[hum]}/")
        and meas_nr[hum] > 0
    ):
        acq_path = f"/mnt/scanner/siemens/{folder_type}/{hum_nr[hum]}/"

        # dicom files
        dicom_path = f"{acq_path}/dicom/"

        if not os.path.exists(dicom_path):
            print("Oops!  Path to dicom files not found.  Check the path name again...")
            break

        # dicom file names
        if meas_nr[hum] < 10:
            dicom_nr = f"{dicom_prefix}-{hum_nr[hum].split('_')[1]}-000{meas_nr[hum]}-"
        else:
            dicom_nr = f"{dicom_prefix}-{hum_nr[hum].split('_')[1]}-00{meas_nr[hum]}-"

        if meas_nr[hum] < 10:
            inv_dicom_nr = (
                f"{dicom_prefix}-{hum_nr[hum].split('_')[1]}-000{inv_meas_nr[hum]}-"
            )
        else:
            inv_dicom_nr = (
                f"{dicom_prefix}-{hum_nr[hum].split('_')[1]}-00{inv_meas_nr[hum]}-"
            )

        # Create separate folder for differenct slice packages
        out_path = f"{data_path}/raw/{hum_nr[hum]}/"

        def ConvertNifti(dicom_files, out_filename, out_path):
            # check existence of empty directory in the study folder
            if not os.path.exists(f"{data_path}/raw/{hum_nr[hum]}/dicom"):
                # make empty directory
                os.makedirs(f"{data_path}/raw/{hum_nr[hum]}/dicom")

            # copy dicom files into the save path folder
            for file in glob.glob(f"{dicom_path}/{dicom_files}*"):
                shutil.copy(file, f"{out_path}/dicom/")

            print("Dicom to nifti converion...")
            # dcm2niix dicom to NIfTI conversion
            convert = Dcm2niix(
                source_dir=f"{out_path}/dicom/",
                output_dir=out_path,
                out_filename=out_filename,
            )
            convert.run()

            # remove temporary files
            shutil.rmtree(f"{out_path}/dicom")

        # convert dicom to nifti
        ConvertNifti(dicom_nr, "dwi", out_path)
        ConvertNifti(inv_dicom_nr, "dwi_inv", out_path)

        # bval files
        if not os.path.exists(f"{out_path}/{bval_file}"):
            print("Oops!  bvalue file not found.  Check the file name again...")
            break
        # bvec files
        if not os.path.exists(f"{out_path}/{bvec_file}"):
            print("Oops!  bvector file not found.  Check the file name again...")
            break

        # take b0 mean
        eroi = fsl.ExtractROI(
            in_file=f"{out_path}/dwi.nii.gz",
            roi_file=f"{out_path}/b0.nii.gz",
            t_min=0,
            t_size=num_b0,
        )
        eroi.run()

        b0mean = fsl.MeanImage(
            in_file=f"{out_path}/b0.nii.gz",
            dimension="T",
            nan2zeros=True,
            out_file=f"{out_path}/dwi_b0_mean.nii.gz",
        )
        b0mean.run()
        # remove extra b0 file
        os.remove(f"{out_path}/b0.nii.gz")

        # ### Denoising
        print("Denoising...")
        # load bvals
        bvals = np.loadtxt(f"{out_path}/{bval_file}")
        # ------------------------------------------------
        #         DIPY Patch2Self denoising
        # ------------------------------------------------
        # load dwi data
        img = nib.load(f"{out_path}/dwi.nii.gz")
        a, b, c, d = img.shape
        data = img.get_fdata()
        # denoising
        denoised = patch2self(data, bvals)
        # save denoised data
        nib.save(
            nib.Nifti1Image(denoised, img.affine), f"{out_path}/dwi_denoised.nii.gz"
        )

        # ### TOPUP and EDDY Correction
        print("TOPUP...")
        # extract b0 image
        fslroi = fsl.ExtractROI(
            in_file=f"{out_path}/dwi_denoised.nii.gz",
            roi_file=f"{out_path}/epi_b0.nii.gz",
            t_min=0,
            t_size=1,
        )
        fslroi.run()
        # prepare files for TOPUP
        merger = fsl.Merge(
            in_files=[f"{out_path}/epi_b0.nii.gz", f"{out_path}/dwi_inv.nii.gz"],
            dimension="t",
            output_type="NIFTI_GZ",
            merged_file=f"{out_path}/epi_b0_merged.nii.gz",
        )
        merger.run()

        # encoding file
        file = open(f"{out_path}/topup_encoding.txt", "w")
        file.write("0 1 0 0.062\n0 -1 0 0.062")
        file.close()

        # config file
        file = open(f"{out_path}/b02b0.cnf", "w")
        line1 = "# Resolution (knot-spacing) of warps in mm"
        line2 = "--warpres=20,16,14,12,10,6,4,4,4"
        line3 = "# Subsampling level (a value of 2 indicates that a 2x2x2 neighbourhood is collapsed to 1 voxel)"
        line4 = "--subsamp=1,1,1,1,1,1,1,1,1"
        line5 = "# FWHM of gaussian smoothing"
        line6 = "--fwhm=8,6,4,3,3,2,1,0,0"
        line7 = "# Maximum number of iterations"
        line8 = "--miter=5,5,5,5,5,10,10,20,20"
        line9 = "# Relative weight of regularisation"
        line10 = "--lambda=0.005,0.001,0.0001,0.000015,0.000005,0.0000005,0.00000005,0.0000000005,0.00000000001"
        line11 = "# If set to 1 lambda is multiplied by the current average squared difference"
        line12 = "--ssqlambda=1"
        line13 = "# Regularisation model"
        line14 = "--regmod=bending_energy"
        line15 = "# If set to 1 movements are estimated along with the field"
        line16 = "--estmov=1,1,1,1,1,0,0,0,0"
        line17 = "# 0=Levenberg-Marquardt, 1=Scaled Conjugate Gradient"
        line18 = "--minmet=0,0,0,0,0,1,1,1,1"
        line19 = "# Quadratic or cubic splines"
        line20 = "--splineorder=3"
        line21 = "# Precision for calculation and storage of Hessian"
        line22 = "--numprec=double"
        line23 = "# Linear or spline interpolation"
        line24 = "--interp=spline"
        line25 = "# If set to 1 the images are individually scaled to a common mean intensity "
        line26 = "--scale=1"
        file.write(
            f"{line1}\n{line2}\n{line3}\n{line4}\n{line5}\n{line6}\n\
            {line7}\n{line8}\n{line9}\n{line10}\n{line11}\n{line12}\n\
            {line13}\n{line14}\n{line15}\n{line16}\n{line17}\n{line18}\n\
            {line19}\n{line20}\n{line21}\n{line22}\n{line23}\n{line24}\n{line25}\n{line26}"
        )
        file.close()

        os.chdir(out_path)
        topup_out_base = "topup_results"
        # TOPUP
        topup = fsl.TOPUP(
            in_file=f"{out_path}/epi_b0_merged.nii.gz",
            encoding_file=f"{out_path}/topup_encoding.txt",
            config=f"{out_path}/b02b0.cnf",
            out_base=f"{out_path}/{topup_out_base}",
            out_corrected=f"{out_path}/epi_b0_merged.nii.gz",
            out_field=f"{out_path}/epi_b0_merged_field.nii.gz",
            out_logfile=f"{out_path}/topup_log.txt",
        )
        topup.run()

        # ApplyTOPUP
        applytopup = fsl.ApplyTOPUP(
            in_files=[f"{out_path}/epi_b0.nii.gz", f"{out_path}/dwi_inv.nii.gz"],
            encoding_file=f"{out_path}/topup_encoding.txt",
            in_topup_fieldcoef=f"{out_path}/{topup_out_base}_fieldcoef.nii.gz",
            in_topup_movpar=f"{out_path}/{topup_out_base}_movpar.txt",
            output_type="NIFTI_GZ",
            out_corrected=f"{out_path}/epi_b0_corrected.nii.gz",
        )
        applytopup.run()

        # EDDY
        print("EDDY...")
        # create an index file
        file = open(f"{out_path}/index.txt", "w")
        for i in range(0, d):
            file.write("1 ")
        file.close()

        # # Brian extraction using FSL
        # btr = fsl.BET(
        #     in_file=f"{out_path}/epi_b0_corrected.nii.gz",
        #     frac=0.5,
        #     out_file=f"{out_path}/dwi_brain.nii.gz",
        #     mask=True,
        #     mask_file=f"{out_path}/dwi_brain_mask.nii.gz",
        # )
        # btr.run()

        # Brian extraction using DIPY
        dwi_bet = nib.load(f"{out_path}/epi_b0.nii.gz").get_fdata()
        b0_mask, mask = median_otsu(np.squeeze(dwi_bet), median_radius=2, numpass=1)
        nib.save(
            nib.Nifti1Image(mask.astype(np.float32), img.affine),
            f"{out_path}/dwi_brain_mask.nii.gz",
        )

        # EDDY
        eddy = fsl.Eddy(
            in_file=f"{out_path}/dwi_denoised.nii.gz",
            in_mask=f"{out_path}/dwi_brain_mask.nii.gz",
            in_index=f"{out_path}/index.txt",
            in_acqp=f"{out_path}/topup_encoding.txt",
            in_topup_movpar=f"{out_path}/{topup_out_base}_movpar.txt",
            in_topup_fieldcoef=f"{out_path}/{topup_out_base}_fieldcoef.nii.gz",
            out_base=f"{out_path}/dwi_eddy_corrected",
            in_bvec=f"{out_path}/{bvec_file}",
            in_bval=f"{out_path}/{bval_file}",
            use_cuda=False,
        )
        eddy.run()
