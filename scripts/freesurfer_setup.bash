# Freesurfer setup based on erisone module file
export FS_FREESURFERENV_NO_OUTPUT=
export PS1="\u@\h:\w\$ "
export PATH=$PATH:/data/nimlab/software/freesurfer-v5.3.0/bin
export PATH=$PATH:/data/nimlab/software/freesurfer-v5.3.0/average
export PATH=$PATH:/data/nimlab/software/freesurfer-v5.3.0/data
export PATH=$PATH:/data/nimlab/software/freesurfer-v5.3.0/diffusion
export PATH=$PATH:/data/nimlab/software/freesurfer-v5.3.0/fsafd
export PATH=$PATH:/data/nimlab/software/freesurfer-v5.3.0/fsfast
export PATH=$PATH:/data/nimlab/software/freesurfer-v5.3.0/matlab
export PATH=$PATH:/data/nimlab/software/freesurfer-v5.3.0/mni
export PATH=$PATH:/data/nimlab/software/freesurfer-v5.3.0/sessions
export PATH=$PATH:/data/nimlab/software/freesurfer-v5.3.0/subjects
export PATH=$PATH:/data/nimlab/software/freesurfer-v5.3.0/tktools
export PATH=$PATH:/data/nimlab/software/freesurfer-v5.3.0/trctrain
export LD_LIBRARY_PATH=/data/nimlab/software/freesurfer-v5.3.0/lib
export FREESURFER_HOME=/data/nimlab/software/freesurfer-v5.3.0
set +u
source $FREESURFER_HOME/SetUpFreeSurfer.sh
set -u
