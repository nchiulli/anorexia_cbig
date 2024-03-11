snakemake -pc1 cbig_01_pre_done.txt 
snakemake -pc1 cbig_1002_01_done.txt 

snakemake -pc1 data/cbig_configs/sub-3015_ses-01/sub-3015_ses-01_multiecho_fmrinii.txt

```bash
# Create milti-echo time file from BIDS-formatted json files:
export met_file=${output_dir}/${participant_id}/${participant_id}_bold.multiechotime
if ls ${bids_dir}/${participant_id}/${session_id}/func/*task-rest*_bold.json | grep "echo" 1> /dev/null 2>&1; then
        jq .'EchoTime' $(ls ${bids_dir}/${participant_id}/${session_id}/func/*task-rest*echo*_bold.json | grep "run-01") | pr -ts, --column ${echo_num} > $met_file
        export met_val=$(cat $met_file)
	echo "Used ${bids_dir}/${participant_id}/${session_id}/func/task-rest_echo_bold.json for multi-echo time information"
```

```
whatis("Loads freesurfer_7.2 environment")
setenv("FREESURFER_HOME","/apps/lib/freesurfer/7.2")
setenv("FIX_VERTEX_AREA","")
setenv("FMRI_ANALYSIS_DIR","/apps/lib/freesurfer/7.2/fsfast")
setenv("FREESURFER","/apps/lib/freesurfer/7.2")
setenv("FSFAST_HOME","/apps/lib/freesurfer/7.2/fsfast")
setenv("FSF_OUTPUT_FORMAT","nii.gz")
setenv("FS_OVERRIDE","0")
setenv("LOCAL_DIR","/apps/lib/freesurfer/7.2/local")
setenv("MINC_BIN_DIR","/apps/lib/freesurfer/7.2/mni/bin")
setenv("MINC_LIB_DIR","/apps/lib/freesurfer/7.2/mni/lib")
setenv("MNI_DATAPATH","/apps/lib/freesurfer/7.2/mni/data")
setenv("MNI_DIR","/apps/lib/freesurfer/7.2/mni")
setenv("MNI_PERL5LIB","/apps/lib/freesurfer/7.2/mni/share/perl5")
setenv("OS","Linux")
prepend_path("PATH","/apps/lib/freesurfer/7.2/mni/bin")
prepend_path("PATH","/apps/lib/freesurfer/7.2/tktools")
prepend_path("PATH","/apps/lib/freesurfer/7.2/fsfast/bin")
prepend_path("PATH","/apps/lib/freesurfer/7.2/bin")
prepend_path("PATH","/apps/lib/freesurfer/7.2/MCRv84/bin")
prepend_path("PATH","/apps/lib/freesurfer/7.2/MCRv84/bin/glnxa64")
prepend_path("LD_LIBRARY_PATH","/apps/lib/freesurfer/7.2/MCRv84/bin/glnxa64")
setenv("MCR_DIR","/apps/lib/freesurfer/7.2/MCRv84")
setenv("PERL5LIB","/apps/lib/freesurfer/7.2/mni/share/perl5")
setenv("SUBJECTS_DIR","/apps/lib/freesurfer/7.2/subjects")
```

```
snakemake -pc8 \
cbig_sub-3015_ses-01_multiecho_done.txt \
cbig_sub-3016_ses-01_multiecho_done.txt \
cbig_sub-3017_ses-01_multiecho_done.txt \
cbig_sub-3017_ses-02_multiecho_done.txt \
cbig_sub-3018_ses-02_multiecho_done.txt \
cbig_sub-3020_ses-01_multiecho_done.txt \
cbig_sub-3022_ses-01_multiecho_done.txt \
cbig_sub-4001_ses-01_multiecho_done.txt \
cbig_sub-4003_ses-01_multiecho_done.txt
```


```bash
mri_vol2vol --mov $input --s sub-3015_ses01 --targ $FREESURFER_HOME/average/mni305.cor.mgz --m3z talairach.m3z --o $tmp_output --no-save-reg --interp cubic

Step 1 - dst to fs:
setenv SUBJECTS_DIR /data/nimlab/software/CBIG_nimlab/CBIG/data/templates/volume/
mri_vol2vol --mov data/cbig_output/sub-3015_ses-01/3015/vol/norm_MNI152_1mm.nii.gz --targ /data/nimlab/software/CBIG_nimlab/CBIG/data/templates/volume/FSL_MNI152_FS4.5.0/mri/norm.nii.gz --s FSL_MNI152_FS4.5.0 --m3z talairach.m3z --o fstest3.nii.gz --no-save-reg --interp cubic

Step 2 - fs to src:
setenv SUBJECTS_DIR /data/nimlab/USERS/cl20/cbig_workflow/data/fs_subjects
export SUBJECTS_DIR=/data/nimlab/USERS/cl20/cbig_workflow/data/fs_subjects
cd /data/nimlab/USERS/cl20/cbig_workflow/data/fs_subjects/sub-3015_ses-01/mri/
mri_vol2vol --mov /data/nimlab/USERS/cl20/cbig_workflow/data/fs_subjects/sub-3015_ses-01/mri/norm.mgz --s sub-3015_ses-01 --targ /data/nimlab/USERS/cl20/cbig_workflow/fstest3.nii.gz --o /data/nimlab/USERS/cl20/cbig_workflow/fs2test1.nii.gz --no-save-reg --interp cubic --inv-morph


```