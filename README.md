Notes:
- The setup files should be sourced in a csh shell
- Even though snakemake is run in a csh shell, it still calls commands with bash. Note that in the cbig rule, the path expansions use $PWD which is valid bash but is not valid csh. 

## Data Prep
Data should be in BIDS format under a folder 'data' inside the anorexia_cbig directory

## Run the pipeline!
The pipeline is coordinated using [Snakemake](https://github.com/snakemake/snakemake) which automatically calculates what steps need to be run in order to produce a desired file. Since we want to produce the target coordinates, we tell Snakemake to produce `data/target/sub-00_ses-pre/sub_target_primary.txt`

data/target/sub-00_ses-pre/sub_target_backup/txt can also be produced; this will use a more conservative brain mask to exclude coordinates where TMS may be too uncomfortable. See masks under the standard_data folder

 ## Set up scripts
 Before running, set env paths in CBIG_setup_eristwo.csh file
 
```
csh
source scripts/CBIG_setup_eristwo.csh
source scripts/freesurfer_setup.csh
```
### The -np flag indicates that this is a dry run. Check that the pipeline steps look like what you want before proceeding
```
snakemake -np data/target/sub-00_ses-pre/sub_target_primary.txt
```
### If everything seems fine, run it with c1 to specify 1 parallel job. Using more is possible if running multiple subjects, but may overload the node.
```
snakemake -pc1 data/target/sub-00_ses-pre/sub_target.txt data/qc/sub-00_ses-pre/seed_corrs.txt
```
