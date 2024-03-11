import os


def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return template, outtype, annotation_classes


def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong where

    allowed template fields - follow python string module:

    item: index within category
    subject: participant id
    seqitem: run number during scanning
    subindex: sub index within group
    """

    bold = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-rest_run-{item:02d}_bold')
    t1 = create_key('sub-{subject}/{session}/anat/sub-{subject}_{session}_T1w')
    fmap_ap = create_key('sub-{subject}/{session}/fmap/sub-{subject}_{session}_dir-AP_epi')
    fmap_pa = create_key('sub-{subject}/{session}/fmap/sub-{subject}_{session}_dir-PA_epi')

    info = {bold: [], t1: [], fmap_ap: [], fmap_pa: []}
    last_run = len(seqinfo)

    for s in seqinfo:
        """
        The namedtuple `s` contains the following fields:

        * total_files_till_now
        * example_dcm_file
        * series_id
        * dcm_dir_name
        * unspecified2
        * unspecified3
        * dim1
        * dim2
        * dim3
        * dim4
        * TR
        * TE
        * protocol_name
        * is_motion_corrected
        * is_derived
        * patient_id
        * study_description
        * referring_physician_name
        * series_description
        * image_type
        """
        if s.protocol_name == "CMRR_MB4_TE4_2.5mm_AP_Rest1":
            info[bold].append(s.series_id)
        if s.series_id == "2-t1_mprage_sag_p2":
            info[t1].append(s.series_id)
        if s.series_id == "3-SpinEchoFieldMap_AP":
            info[fmap_ap].append(s.series_id)
        if s.series_id == "4-SpinEchoFieldMap_PA":
            info[fmap_pa].append(s.series_id)
    return info
