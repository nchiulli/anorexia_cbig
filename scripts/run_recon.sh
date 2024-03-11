# $1: fs_subjects directory
# $2: T1 image
# $3: Subject label
module load gcc/9.3.0
module load freesurfer/7.2.0
export SUBJECTS_DIR=`readlink -f $1`
export FS_LICENSE=`readlink -f ./license.txt`
# bsub -q big-multi -n 4 -R 'rusage[mem=16000]' -u clin@bwh.harvard.edu recon-all -s $3 -i $2 -all -parallel -openmp 4
recon-all -s $3 -i $2 -all -parallel -openmp 8
