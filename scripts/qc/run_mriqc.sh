module load singularity/3.7.0
singularity run -B $1:/data:ro -B $2:/out /data/nimlab/singularity_containers/mriqc.sif  \
    /data /out participant --participant_label $3