import json
from datetime import datetime
import connectomics as cs
import functions as fn
import datasets as ds
from nilearn import image, maskers
from numpy.linalg import inv
from scipy import stats
import numpy as np
import os
import re

# Set rule order to prioritize Spatial Distortion Correction unless fieldmaps unavailable
ruleorder: cbig_multiecho_sdr > cbig_multiecho > cbig_sdr_magphase > cbig
# ruleorder: cbig_multiecho_sdr > cbig_multiecho > cbig_sdr_magphase
ruleorder: concat_tcs_multi > concat_tcs_single


def determine_first_run(sub, ses):
    base_path = f"data/BIDS/sub-{sub}/ses-{ses}/func/sub-{sub}_ses-{ses}_task-rest"
    
    # Define a helper function to check the existence of files
    def file_exists(run):
        no_echo_file = f"{base_path}_{run}_bold.json"
        echo_file = f"{base_path}_{run}_echo-1_bold.json"
        return os.path.exists(no_echo_file) or os.path.exists(echo_file)

    # Check for run-01 file
    if file_exists("run-01"):
        return "run-01"

    # If run-01 doesn't exist, check for run-02 file
    if file_exists("run-02"):
        return "run-02"

    # If neither run-01 nor run-02 exist, return run-03
    return "run-03"


# Run freesurfer recon-all, which is a prerequisite to CBIG
rule recon_all:
    input:
        "data/BIDS/sub-{sub}/ses-{ses}/anat/sub-{sub}_ses-{ses}_T1w.nii.gz"
    output:
        directory("data/fs_subjects/sub-{sub}_ses-{ses}"),
        "data/fs_subjects/sub-{sub}_ses-{ses}/scripts/recon-all.done"
    run:
        # Snakemake auto-creates output dirs, but freesurfer fails if output dirs exist
        shell(f"rm -rf data/fs_subjects/sub-{wildcards.sub}_ses-{wildcards.ses}/")
        # Don't destroy our node
        shell(f"nice -n 19 bash scripts/run_recon.sh data/fs_subjects {input} sub-{wildcards.sub}_ses-{wildcards.ses}")

rule recon_all_butler:
    input:
        "data/BIDS/sub-{sub}/ses-{ses}/anat/sub-{sub}_ses-{ses}_e2_T1w.nii.gz"
    output:
        directory("data/fs_subjects/sub-{sub}_ses-{ses}"),
        "data/fs_subjects/sub-{sub}_ses-{ses}/scripts/recon-all.done"
    run:
        # Snakemake auto-creates output dirs, but freesurfer fails if output dirs exist
        shell(f"rm -rf data/fs_subjects/sub-{wildcards.sub}_ses-{wildcards.ses}/")
        # Don't destroy our node
        shell(f"nice -n 19 bash scripts/run_recon.sh data/fs_subjects {input} sub-{wildcards.sub}_ses-{wildcards.ses}")


# Prepare and format misc files needed by CBIG
rule create_fmrinii:
    input:
        "data/BIDS/sub-{sub}/ses-{ses}"
    output:
        "data/cbig_configs/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_fmrinii.txt"
    run:
        shell(f"mkdir -p data/cbig_configs/sub-{wildcards.sub}_ses-{wildcards.ses}")
        basepath = f"{input}/func/sub-{wildcards.sub}_ses-{wildcards.ses}"
        #print(basepath)
        runs = glob_wildcards(basepath + "_task-rest_run-{run}_bold.nii.gz").run
        with open(output[0],"w+") as f:
            for r in runs:
                f.write(f"{r} {basepath}_task-rest_run-{r}_bold.nii.gz \n")

rule create_multiecho_fmrinii:
    # input:
    #     "data/BIDS/sub-{sub}/ses-{ses}",
    #     "data/BIDS/sub-{sub}/ses-{ses}/func/sub-{sub}_ses-{ses}_task-rest_run-01_echo-1_bold.json"
    input:
        lambda wildcards: f"data/BIDS/sub-{wildcards.sub}/ses-{wildcards.ses}",
        lambda wildcards: f"data/BIDS/sub-{wildcards.sub}/ses-{wildcards.ses}/func/sub-{wildcards.sub}_ses-{wildcards.ses}_task-rest_{determine_first_run(wildcards.sub, wildcards.ses)}_echo-1_bold.json"
    output:
        "data/cbig_configs/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_multiecho_fmrinii.txt"
    run:
        shell(f"mkdir -p data/cbig_configs/sub-{wildcards.sub}_ses-{wildcards.ses}")
        basepath = f"{input[0]}/func/sub-{wildcards.sub}_ses-{wildcards.ses}"
        runs, echos = glob_wildcards(basepath + "_task-rest_run-{run}_echo-{echo}_bold.nii.gz")
        run_echos = {}
        for i in zip(runs, echos):
            if i[0] not in run_echos:
                run_echos[i[0]] = [i[1]]
                continue
            run_echos[i[0]].append(i[1])
        with open(output[0],"w+") as f:
            for r in run_echos:
                line = str(r)
                for e in run_echos[r]:
                    line += f" {basepath}_task-rest_run-{r}_echo-{e}_bold.nii.gz"
                f.write(line + "\n")

rule create_st_file:
    input:
        "data/BIDS/sub-{sub}/ses-{ses}"
    output:
        "data/cbig_configs/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_slicetiming.txt"
    run:
        shell("python scripts/reformat_slicetiming.py -d data/BIDS -p sub-{wildcards.sub} -s ses-{wildcards.ses} -o {output[0]}")

rule create_multiecho_st_files:
    # input:
    #     "data/BIDS/sub-{sub}/ses-{ses}",
    #     "data/BIDS/sub-{sub}/ses-{ses}/func/sub-{sub}_ses-{ses}_task-rest_run-01_echo-1_bold.json"
    input:
        lambda wildcards: f"data/BIDS/sub-{wildcards.sub}/ses-{wildcards.ses}",
        lambda wildcards: f"data/BIDS/sub-{wildcards.sub}/ses-{wildcards.ses}/func/sub-{wildcards.sub}_ses-{wildcards.ses}_task-rest_{determine_first_run(wildcards.sub, wildcards.ses)}_echo-1_bold.json"
    output:
        directory("data/cbig_configs/sub-{sub}_ses-{ses}/multiecho_slicetimings")
    run:
        basepath = f"{input[0]}/func/sub-{wildcards.sub}_ses-{wildcards.ses}"
        runs, echos = glob_wildcards(basepath + "_task-rest_run-{run}_echo-{echo}_bold.nii.gz")
        shell("mkdir -p {output[0]}")
        for e in set(echos):
            shell("python scripts/reformat_slicetiming_multiecho.py -d data/BIDS -p sub-{wildcards.sub} -s ses-{wildcards.ses} -o {output[0]}/sub-{wildcards.sub}_ses-{wildcards.ses}_echo-{e}_slicetiming.txt -e {e}")

rule create_echotimes_file:
    input:
        "data/BIDS/sub-{sub}/ses-{ses}"
    output:
        "data/cbig_configs/sub-{sub}_ses-{ses}/echotimes.txt"
    run:
        basepath = f"{input[0]}/func/sub-{wildcards.sub}_ses-{wildcards.ses}"
        first_run = determine_first_run(wildcards.sub, wildcards.ses)
        echos = glob_wildcards(f"{basepath}_task-rest_{first_run}_echo-{{echo}}_bold.nii.gz")[0]
        echotimes = []
        for e in echos:
            with open(f"{basepath}_task-rest_{first_run}_echo-{e}_bold.json") as f:
                sidecar = json.load(f)
                echotimes.append(str(round(sidecar['EchoTime'] * 1000,2)))
        with open(output[0],"w+") as f:
            f.write(",".join(echotimes))      



# Run CBIG
rule cbig:
    input:
        "data/fs_subjects/sub-{sub}_ses-{ses}",
        "data/cbig_configs/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_fmrinii.txt",
        "data/cbig_configs/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_slicetiming.txt"
    output:
        "cbig_sub-{sub}_ses-{ses}_singleecho_done.txt",
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/norm_MNI152_1mm.nii.gz"
    run:
        begin_time = datetime.now()
        shell("mkdir -p data/cbig_output/sub-{wildcards.sub}_ses-{wildcards.ses}/")
        shell("export st_file=$PWD/{input[2]} && \
                export met_val='' && \
                \
                source scripts/freesurfer_setup.bash && \
                echo $FREESURFER_HOME && \
                nice -n 19 csh -c \"$CBIG_CODE_DIR/stable_projects/preprocessing/CBIG_fMRI_Preproc2016/CBIG_preproc_fMRI_preprocess.csh \
                -s {wildcards.sub} \
                -fmrinii $PWD/{input[1]} \
                -anat_s sub-{wildcards.sub}_ses-{wildcards.ses} \
                -anat_d $PWD/data/fs_subjects \
                -output_d $PWD/data/cbig_output/sub-{wildcards.sub}_ses-{wildcards.ses} \
                -config $PWD/scripts/CBIG_preproc_fMRI_preprocess.config\"")
        end_time = datetime.now()
        begin_time_str = begin_time.strftime("%d/%m/%Y %H:%M:%S")
        end_time_str = end_time.strftime("%d/%m/%Y %H:%M:%S")
        with open(f"cbig_sub-{wildcards.sub}_ses-{wildcards.ses}_singleecho_done.txt", "w+") as f:
            f.write(f"CBIG started {begin_time_str} \n")
            f.write(f"CBIG ended {end_time_str} \n")
            f.write(f"CBIG took {(end_time - begin_time).total_seconds() / 60} minutes")

rule cbig_sdr_magphase:
    input:
        "data/fs_subjects/sub-{sub}_ses-{ses}",
        "data/cbig_configs/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_fmrinii.txt",
        "data/cbig_configs/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_slicetiming.txt",
        "data/BIDS/sub-{sub}/ses-{ses}/fmap/sub-{sub}_ses-{ses}_magnitude1.nii.gz",
        "data/BIDS/sub-{sub}/ses-{ses}/fmap/sub-{sub}_ses-{ses}_phasediff.nii.gz"
    output:
        "cbig_sub-{sub}_ses-{ses}_singleecho_done.txt",
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/norm_MNI152_1mm.nii.gz"
    run:
        sub = wildcards.sub
        ses = wildcards.ses
        begin_time = datetime.now()
        with open(f"data/BIDS/sub-{sub}/ses-{ses}/fmap/sub-{sub}_ses-{ses}_phasediff.json") as f:
            data = json.load(f)
            delta_val = (data["EchoTime2"] - data["EchoTime1"]) * 1000
        with open(f"data/BIDS/sub-{sub}/ses-{ses}/func/sub-{sub}_ses-{ses}_task-rest_{determine_first_run(sub, ses)}_bold.json") as f:
            ees = json.load(f)["EffectiveEchoSpacing"] * 1000
        with open(f"data/BIDS/sub-{sub}/ses-{ses}/func/sub-{sub}_ses-{ses}_task-rest_{determine_first_run(sub, ses)}_bold.json") as f:
            te = json.load(f)["EchoTime"] * 1000

        mag_path = os.path.abspath(f"data/BIDS/sub-{sub}/ses-{ses}/fmap/sub-{sub}_ses-{ses}_magnitude1.nii.gz")
        phasediff_path = os.path.abspath(f"data/BIDS/sub-{sub}/ses-{ses}/fmap/sub-{sub}_ses-{ses}_phasediff.nii.gz")
        shell("mkdir -p data/cbig_output/sub-{sub}_ses-{ses}/")
        shell("export st_file=$PWD/{input[2]} && \
                export met_val='' && \
                \
                export mag_image_path={mag_path} && \
                export phasediff_image_path={phasediff_path} && \
                export delta_val={delta_val} && \
                export ees={ees} && \
                export te={te} && \
                \
                source scripts/freesurfer_setup.bash && \
                echo $FREESURFER_HOME && \
                nice -n 19 csh -c \"$CBIG_CODE_DIR/stable_projects/preprocessing/CBIG_fMRI_Preproc2016/CBIG_preproc_fMRI_preprocess.csh \
                -s {wildcards.sub} \
                -fmrinii $PWD/{input[1]} \
                -anat_s sub-{wildcards.sub}_ses-{wildcards.ses} \
                -anat_d $PWD/data/fs_subjects \
                -output_d $PWD/data/cbig_output/sub-{wildcards.sub}_ses-{wildcards.ses} \
                -config scripts/BWH_sdr_magphase.config\"")
        end_time = datetime.now()
        begin_time_str = begin_time.strftime("%d/%m/%Y %H:%M:%S")
        end_time_str = end_time.strftime("%d/%m/%Y %H:%M:%S")
        with open(f"cbig_sub-{wildcards.sub}_ses-{wildcards.ses}_singleecho_done.txt", "w+") as f:
            f.write(f"CBIG started {begin_time_str} \n")
            f.write(f"CBIG ended {end_time_str} \n")
            f.write(f"CBIG took {(end_time - begin_time).total_seconds() / 60} minutes")

rule cbig_multiecho_sdr:
    input:
        "data/fs_subjects/sub-{sub}_ses-{ses}",
        "data/cbig_configs/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_multiecho_fmrinii.txt",
        "data/cbig_configs/sub-{sub}_ses-{ses}/multiecho_slicetimings",
        "data/cbig_configs/sub-{sub}_ses-{ses}/echotimes.txt",
        "data/BIDS/sub-{sub}/ses-{ses}/fmap/sub-{sub}_ses-{ses}_dir-AP_epi.nii.gz",
        "data/BIDS/sub-{sub}/ses-{ses}/fmap/sub-{sub}_ses-{ses}_dir-PA_epi.nii.gz"
    output:
        "cbig_sub-{sub}_ses-{ses}_multiecho_done.txt",
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/norm_MNI152_1mm.nii.gz"
    run:
        sub = wildcards.sub
        ses = wildcards.ses
        begin_time = datetime.now()
        with open(f"data/BIDS/sub-{sub}/ses-{ses}/fmap/sub-{sub}_ses-{ses}_dir-AP_epi.json") as f:
            ap_trt = json.load(f)["TotalReadoutTime"]
        with open(f"data/BIDS/sub-{sub}/ses-{ses}/fmap/sub-{sub}_ses-{ses}_dir-PA_epi.json") as f:
            pa_trt = json.load(f)["TotalReadoutTime"]
        with open(f"data/BIDS/sub-{sub}/ses-{ses}/func/sub-{sub}_ses-{ses}_task-rest_{determine_first_run(sub, ses)}_echo-1_bold.json") as f:
            ees = json.load(f)["EffectiveEchoSpacing"] * 1000
        with open(f"data/BIDS/sub-{sub}/ses-{ses}/func/sub-{sub}_ses-{ses}_task-rest_{determine_first_run(sub, ses)}_echo-1_bold.json") as f:
            te = json.load(f)["EchoTime"] * 1000

        ap_path = os.path.abspath(f"data/BIDS/sub-{sub}/ses-{ses}/fmap/sub-{sub}_ses-{ses}_dir-AP_epi.nii.gz")
        pa_path = os.path.abspath(f"data/BIDS/sub-{sub}/ses-{ses}/fmap/sub-{sub}_ses-{ses}_dir-PA_epi.nii.gz")
        shell("mkdir -p data/cbig_output/sub-{sub}_ses-{ses}/")
        shell("export st_files=$(ls -p $PWD/{input[2]}/* | tr '\n' ',') && \
                export met_val=$(cat {input[3]}) && \
                \
                export j_minus_image_path={ap_path} && \
                export j_plus_image_path={pa_path} && \
                export j_minus_trt={ap_trt} && \
                export j_plus_trt={pa_trt} && \
                export ees={ees} && \
                export te={te} && \
                \
                source scripts/freesurfer_setup.bash && \
                echo $FREESURFER_HOME && \
                export CBIG_CODE_DIR=/data/nimlab/software/CBIG_nimlab/CBIG && \
                nice -n 19 csh -c \"$CBIG_CODE_DIR/stable_projects/preprocessing/CBIG_fMRI_Preproc2016/CBIG_preproc_fMRI_preprocess.csh \
                -s {wildcards.sub} \
                -fmrinii $PWD/{input[1]} \
                -anat_s sub-{wildcards.sub}_ses-{wildcards.ses} \
                -anat_d $PWD/data/fs_subjects \
                -output_d $PWD/data/cbig_output/sub-{wildcards.sub}_ses-{wildcards.ses} \
                -config scripts/BWH_multiecho_sdr.config\"")
        end_time = datetime.now()
        begin_time_str = begin_time.strftime("%d/%m/%Y %H:%M:%S")
        end_time_str = end_time.strftime("%d/%m/%Y %H:%M:%S")
        with open(f"cbig_sub-{wildcards.sub}_ses-{wildcards.ses}_multiecho_done.txt", "w+") as f:
            f.write(f"CBIG started {begin_time_str} \n")
            f.write(f"CBIG ended {end_time_str} \n")
            f.write(f"CBIG took {(end_time - begin_time).total_seconds() / 60} minutes")

rule cbig_multiecho:
    input:
        "data/fs_subjects/sub-{sub}_ses-{ses}",
        "data/cbig_configs/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_multiecho_fmrinii.txt",
        "data/cbig_configs/sub-{sub}_ses-{ses}/multiecho_slicetimings",
        "data/cbig_configs/sub-{sub}_ses-{ses}/echotimes.txt"
    output:
        "cbig_sub-{sub}_ses-{ses}_multiecho_done.txt",
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/norm_MNI152_1mm.nii.gz"
    run:
        sub = wildcards.sub
        ses = wildcards.ses
        begin_time = datetime.now()

        shell("mkdir -p data/cbig_output/sub-{sub}_ses-{ses}/")
        shell("export st_files=$(ls -p $PWD/{input[2]}/* | tr '\n' ',') && \
                export met_val=$(cat {input[3]}) && \
                \
                source scripts/freesurfer_setup.bash && \
                echo $FREESURFER_HOME && \
                nice -n 19 csh -c \"$CBIG_CODE_DIR/stable_projects/preprocessing/CBIG_fMRI_Preproc2016/CBIG_preproc_fMRI_preprocess.csh \
                -s {wildcards.sub} \
                -fmrinii $PWD/{input[1]} \
                -anat_s sub-{wildcards.sub}_ses-{wildcards.ses} \
                -anat_d $PWD/data/fs_subjects \
                -output_d $PWD/data/cbig_output/sub-{wildcards.sub}_ses-{wildcards.ses} \
                -config scripts/BWH_multiecho.config\"")
        end_time = datetime.now()
        begin_time_str = begin_time.strftime("%d/%m/%Y %H:%M:%S")
        end_time_str = end_time.strftime("%d/%m/%Y %H:%M:%S")
        with open(f"cbig_sub-{wildcards.sub}_ses-{wildcards.ses}_multiecho_done.txt", "w+") as f:
            f.write(f"CBIG started {begin_time_str} \n")
            f.write(f"CBIG ended {end_time_str} \n")
            f.write(f"CBIG took {(end_time - begin_time).total_seconds() / 60} minutes")

# Concatenate final output timecourses
rule concat_tcs_multi:
    input:
        "cbig_sub-{sub}_ses-{ses}_multiecho_done.txt",
        #"data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol"
    output:
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_concat.nii.gz"
    run:
        shell(f"fslmerge -t {output[0]} data/cbig_output/sub-{wildcards.sub}_ses-{wildcards.ses}/{wildcards.sub}/vol/*_finalmask.nii.gz")

# Concatenate final output timecourses
rule concat_tcs_single:
    input:
        "cbig_sub-{sub}_ses-{ses}_singleecho_done.txt",
        #"data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol"
    output:
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_concat.nii.gz"
    run:
        shell(f"fslmerge -t {output[0]} data/cbig_output/sub-{wildcards.sub}_ses-{wildcards.ses}/{wildcards.sub}/vol/*_finalmask.nii.gz")


# Split concatenated output in half for intra-subject qc purposes
rule split_concat:
    input:
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_concat.nii.gz"
    output:
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_split-1.nii.gz",
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_split-2.nii.gz"
    run:
        concat_img = image.load_img(input[0])
        half = int(concat_img.shape[3] / 2)
        split1 = image.index_img(concat_img, slice(0, half))
        split2 = image.index_img(concat_img, slice(half,None))
        split1.to_filename(output[0])
        split2.to_filename(output[1])


# Calculate connectivity to anorexia map
rule anorexia_conn:
    input:
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_concat.nii.gz",
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_split-1.nii.gz",
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_split-2.nii.gz"
    output:
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_anorexia_connectivity_z.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-1_anorexia_connectivity_z.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-2_anorexia_connectivity_z.nii.gz"
    run:
        res = cs.singlesubject_seed_conn("standard_data/anorexia_network.nii.gz", input[0], transform="zscore")
        res.to_filename(output[0])
        res = cs.singlesubject_seed_conn("standard_data/anorexia_network.nii.gz", input[1], transform="zscore")
        res.to_filename(output[1])
        res = cs.singlesubject_seed_conn("standard_data/anorexia_network.nii.gz", input[2], transform="zscore")
        res.to_filename(output[2])

rule searchlight:
    input:
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_concat.nii.gz",
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_split-1.nii.gz",
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_split-2.nii.gz"
    output:
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_searchlight_primary.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_searchlight_peak_primary.txt",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-1_searchlight_primary.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-1_searchlight_peak_primary.txt",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-2_searchlight_primary.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-2_searchlight_peak_primary.txt",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_searchlight_backup.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_searchlight_peak_backup.txt",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-1_searchlight_backup.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-1_searchlight_peak_backup.txt",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-2_searchlight_backup.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-2_searchlight_peak_backup.txt"
    run:
        shell(f"python scripts/searchlight.py {input[0]} {output[0]} {output[1]} standard_data/primary_target_mask.nii.gz")
        shell(f"python scripts/searchlight.py {input[1]} {output[2]} {output[3]} standard_data/primary_target_mask.nii.gz")
        shell(f"python scripts/searchlight.py {input[2]} {output[4]} {output[5]} standard_data/primary_target_mask.nii.gz")
        shell(f"python scripts/searchlight.py {input[0]} {output[6]} {output[7]} standard_data/backup_target_mask.nii.gz")
        shell(f"python scripts/searchlight.py {input[1]} {output[8]} {output[9]} standard_data/backup_target_mask.nii.gz")
        shell(f"python scripts/searchlight.py {input[2]} {output[10]} {output[11]} standard_data/backup_target_mask.nii.gz")

rule searchlight_spatialcorr:
    input:
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_concat.nii.gz",
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_split-1.nii.gz",
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_split-2.nii.gz"
    output:
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_searchlight_spatialcorr.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-1_searchlight_spatialcorr.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-2_searchlight_spatialcorr.nii.gz"
    run:
        shell(f"python scripts/searchlight_spatialcorr.py {input[0]} {output[0]}")
        shell(f"python scripts/searchlight_spatialcorr.py {input[1]} {output[1]}")
        shell(f"python scripts/searchlight_spatialcorr.py {input[2]} {output[2]}")

rule cluster_cog:
    input:
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_anorexia_connectivity_z.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-1_anorexia_connectivity_z.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-2_anorexia_connectivity_z.nii.gz"
    output:
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_cog.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-1_cog.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-2_cog.nii.gz"
    run:
        shell(f"python scripts/cluster_cog.py {input[0]} {output[0]}")
        shell(f"python scripts/cluster_cog.py {input[1]} {output[1]}")
        shell(f"python scripts/cluster_cog.py {input[2]} {output[2]}")


rule subject_target:
    input:
        "data/fs_subjects/sub-{sub}_ses-{ses}",
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/norm_MNI152_1mm.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_searchlight_peak_primary.txt",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_searchlight_peak_backup.txt"
    output:
        "data/target/sub-{sub}_ses-{ses}/sub_target_primary.txt",
        "data/target/sub-{sub}_ses-{ses}/sub_target_backup.txt"
    run:
        #os.makedirs(f"data/target/sub-{wildcards.sub}_ses-{wildcards.ses}/",exist_ok=True)
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

        # primary target
        ref = image.load_img(input[1])
        with open(input[2]) as f:
            lines = [l.strip() for l in f.readlines()]
        coord_strings = lines[1].split(": ")[-1].replace("(","").replace(")","").split(",")
        print(coord_strings)
        coords = (float(coord_strings[0]), float(coord_strings[1]), float(coord_strings[2])) 
        print(coords)

        blank = np.zeros(ref.shape)
        vox_coords = image.coord_transform(coords[0], coords[1], coords[2], inv(ref.affine))
        mni_sphere = fn.make_sphere(ref, vox_coords, 2, 100)
        mni_sphere.to_filename(f"data/target/sub-{wildcards.sub}_ses-{wildcards.ses}/mni.nii.gz")

        # Transform to FS space
        shell(f"SUBJECTS_DIR=/data/nimlab/software/CBIG_nimlab/CBIG/data/templates/volume/ \
                mri_vol2vol --mov data/target/sub-{wildcards.sub}_ses-{wildcards.ses}/mni.nii.gz\
                --targ /data/nimlab/software/CBIG_nimlab/CBIG/data/templates/volume/FSL_MNI152_FS4.5.0/mri/norm.nii.gz \
                --s FSL_MNI152_FS4.5.0 --m3z talairach.m3z --o data/target/sub-{wildcards.sub}_ses-{wildcards.ses}/fs.nii.gz --no-save-reg --interp cubic")

        # Threshold out interpolation noise
        image.threshold_img(f"data/target/sub-{wildcards.sub}_ses-{wildcards.ses}/fs.nii.gz",10) \
            .to_filename(f"data/target/sub-{wildcards.sub}_ses-{wildcards.ses}/fs_thresh.nii.gz")
        fs_thresh_fullpath = os.path.abspath(f"data/target/sub-{wildcards.sub}_ses-{wildcards.ses}/fs_thresh.nii.gz")
        sub_target_fullpath = os.path.abspath(f"data/target/sub-{wildcards.sub}_ses-{wildcards.ses}/sub.nii.gz")
        shell(f"cd {input[0]}/mri/ && \
                export SUBJECTS_DIR={os.path.abspath('data/fs_subjects')} && \
                mri_vol2vol --mov norm.mgz --s sub-{wildcards.sub}_ses-{wildcards.ses} \
                --targ {fs_thresh_fullpath}  --o {sub_target_fullpath} --no-save-reg --interp cubic --inv-morph")

        sub_transformed = image.threshold_img(sub_target_fullpath,10)
        sub_target_vox = center_of_gravity(sub_transformed.get_fdata())
        sub_target_coord = image.coord_transform(sub_target_vox[0], sub_target_vox[1], sub_target_vox[2], sub_transformed.affine)
        with open(output[0], "w+") as f:
            f.write(f"{sub_target_coord[0]}, {sub_target_coord[1]}, {sub_target_coord[2]}\n")

        # backup target
        ref = image.load_img(input[1])
        with open(input[3]) as f:
            lines = [l.strip() for l in f.readlines()]
        coord_strings = lines[1].split(": ")[-1].replace("(","").replace(")","").split(",")
        print(coord_strings)
        coords = (float(coord_strings[0]), float(coord_strings[1]), float(coord_strings[2])) 
        print(coords)

        blank = np.zeros(ref.shape)
        vox_coords = image.coord_transform(coords[0], coords[1], coords[2], inv(ref.affine))
        mni_sphere = fn.make_sphere(ref, vox_coords, 2, 100)
        mni_sphere.to_filename(f"data/target/sub-{wildcards.sub}_ses-{wildcards.ses}/mni.nii.gz")

        # Transform to FS space
        shell(f"SUBJECTS_DIR=/data/nimlab/software/CBIG_nimlab/CBIG/data/templates/volume/ \
                mri_vol2vol --mov data/target/sub-{wildcards.sub}_ses-{wildcards.ses}/mni.nii.gz\
                --targ /data/nimlab/software/CBIG_nimlab/CBIG/data/templates/volume/FSL_MNI152_FS4.5.0/mri/norm.nii.gz \
                --s FSL_MNI152_FS4.5.0 --m3z talairach.m3z --o data/target/sub-{wildcards.sub}_ses-{wildcards.ses}/fs.nii.gz --no-save-reg --interp cubic")

        # Threshold out interpolation noise
        image.threshold_img(f"data/target/sub-{wildcards.sub}_ses-{wildcards.ses}/fs.nii.gz",10) \
            .to_filename(f"data/target/sub-{wildcards.sub}_ses-{wildcards.ses}/fs_thresh.nii.gz")
        fs_thresh_fullpath = os.path.abspath(f"data/target/sub-{wildcards.sub}_ses-{wildcards.ses}/fs_thresh.nii.gz")
        sub_target_fullpath = os.path.abspath(f"data/target/sub-{wildcards.sub}_ses-{wildcards.ses}/sub.nii.gz")
        shell(f"cd {input[0]}/mri/ && \
                export SUBJECTS_DIR={os.path.abspath('data/fs_subjects')} && \
                mri_vol2vol --mov norm.mgz --s sub-{wildcards.sub}_ses-{wildcards.ses} \
                --targ {fs_thresh_fullpath}  --o {sub_target_fullpath} --no-save-reg --interp cubic --inv-morph")

        sub_transformed = image.threshold_img(sub_target_fullpath,10)
        sub_target_vox = center_of_gravity(sub_transformed.get_fdata())
        sub_target_coord = image.coord_transform(sub_target_vox[0], sub_target_vox[1], sub_target_vox[2], sub_transformed.affine)
        with open(output[1], "w+") as f:
            f.write(f"{sub_target_coord[0]}, {sub_target_coord[1]}, {sub_target_coord[2]}\n")

rule window_clust:
    input:
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_concat.nii.gz",
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_split-1.nii.gz",
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_split-2.nii.gz"
    output:
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_windowclust.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-1_windowclust.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-2_windowclust.nii.gz"
    run:
        shell(f"python scripts/window_clust.py {input[0]} {output[0]}")
        shell(f"python scripts/window_clust.py {input[1]} {output[1]}")
        shell(f"python scripts/window_clust.py {input[2]} {output[2]}")

# QC metrics
rule mriqc:
    input:
        "data/BIDS/sub-{sub}/ses-{ses}/"
    output:
        directory("data/qc/sub-{sub}_ses-{ses}/mriqc/"),
    run:
        shell("mkdir -p data/qc/sub-{wildcards.sub}_ses-{wildcards.ses}/mriqc/")
        shell("sh scripts/qc/run_mriqc.sh data/BIDS data/qc/sub-{wildcards.sub}_ses-{wildcards.ses}/mriqc/ {wildcards.sub}")

rule qc_seeds:
    input:
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_concat.nii.gz"
    output:
        "data/qc/sub-{sub}_ses-{ses}/seed_corrs.txt"
    run:
        masker = maskers.NiftiMasker(mask_img=ds.get_img("MNI152_T1_2mm_brain_mask")).fit()

        m1_map = cs.singlesubject_seed_conn("standard_data/M1_weighted_ROI_harvox.nii.gz", input[0], transform='zscore')
        m1_map_std = image.load_img("standard_data/M1_weighted_ROI_harvox_AvgR_Fz.nii.gz")
        m1_corr = stats.pearsonr(masker.transform(m1_map)[0,:],masker.transform(m1_map_std)[0,:])[0]

        pcc_map = cs.singlesubject_seed_conn("standard_data/PCC_weighted_ROI_harvox.nii.gz", input[0], transform='zscore')
        pcc_map_std = image.load_img("standard_data/PCC_weighted_ROI_harvox_AvgR_Fz.nii.gz")
        pcc_corr = stats.pearsonr(masker.transform(pcc_map)[0,:],masker.transform(pcc_map_std)[0,:])[0]

        sg_map = cs.singlesubject_seed_conn("standard_data/SG_6_16_-10_r10_msk3.nii.gz", input[0], transform='zscore')
        sg_map_std = image.load_img("standard_data/SG_6_16_-10_r10_msk3_AvgR_Fz.nii.gz")
        sg_corr = stats.pearsonr(masker.transform(sg_map)[0,:],masker.transform(sg_map_std)[0,:])[0]

        with open(output[0],"w+") as f:
            f.write(f"M1: {m1_corr}\n")
            f.write(f"PCC: {pcc_corr}\n")
            f.write(f"SG: {sg_corr}\n")

rule split_corr:
    input:
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_split-1.nii.gz",
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_split-2.nii.gz"
    output:
        "data/qc/sub-{sub}_ses-{ses}/split_corrs.txt"
    run:
        masker = maskers.NiftiMasker(mask_img=ds.get_img("MNI152_T1_2mm_brain_mask")).fit()

        m1_map_1 = cs.singlesubject_seed_conn("standard_data/M1_weighted_ROI_harvox.nii.gz", input[0], transform='zscore')
        m1_map_2 = cs.singlesubject_seed_conn("standard_data/M1_weighted_ROI_harvox.nii.gz", input[1], transform='zscore')
        m1_corr = stats.pearsonr(masker.transform(m1_map_1)[0,:],masker.transform(m1_map_2)[0,:])[0]

        pcc_map_1 = cs.singlesubject_seed_conn("standard_data/PCC_weighted_ROI_harvox.nii.gz", input[0], transform='zscore')
        pcc_map_2 = cs.singlesubject_seed_conn("standard_data/PCC_weighted_ROI_harvox.nii.gz", input[1], transform='zscore')
        pcc_corr = stats.pearsonr(masker.transform(pcc_map_1)[0,:],masker.transform(pcc_map_2)[0,:])[0]

        sg_map_1 = cs.singlesubject_seed_conn("standard_data/SG_6_16_-10_r10_msk3.nii.gz", input[0], transform='zscore')
        sg_map_2 = cs.singlesubject_seed_conn("standard_data/SG_6_16_-10_r10_msk3.nii.gz", input[1], transform='zscore')
        sg_corr = stats.pearsonr(masker.transform(sg_map_1)[0,:],masker.transform(sg_map_2)[0,:])[0]

        with open(output[0],"w+") as f:
            f.write(f"M1_split: {m1_corr}\n")
            f.write(f"PCC_split: {pcc_corr}\n")
            f.write(f"SG_split: {sg_corr}\n")
