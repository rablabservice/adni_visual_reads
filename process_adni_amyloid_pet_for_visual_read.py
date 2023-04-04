#!/Users/dschonhaut/mambaforge/envs/nipy310/bin/python

"""
spm.py

Description: Functions for spm12 coregistration, normalization, realignment,
smoothing, and segmentation, with the ability to pass input arguments
from the command line.
"""

import sys
import os
import os.path as op
import argparse
from glob import glob
import datetime
import numpy as np
import pandas as pd

sys.path.append(op.join(op.expanduser("~"), "code"))
import general.nifti.nifti_ops as nops
import general.nifti.spm as spm
from general.basic.helper_funcs import Timer


def _get_proc_df(proj_dir, process_subjs=None, overwrite=False):
    """Get a dataframe of subjects to process."""
    # Get a list of subjects to process.
    subjs = os.listdir(op.join(proj_dir, "data", "raw"))

    # Create a dataframe of subjects to process.
    proc_df = pd.DataFrame(
        index=subjs,
        columns=[
            "pet_date",
            "tracer",
            "raw_petf",
            "proc_petf",
            "raw_res",
            "proc_res",
            "process",
            "processing_complete",
            "notes",
        ],
    )
    proc_df["proc_res"] = [[8, 8, 8]] * len(proc_df)
    proc_df["process"] = True
    proc_df["processing_complete"] = False
    proc_df["notes"] = ""
    if process_subjs:
        exclude_subjs = [subj for subj in proc_df.index if subj not in process_subjs]
        proc_df.loc[exclude_subjs, "process"] = False

    # Find the raw PET scan for each subject.
    for subj in subjs:
        raw_petfs = glob(
            op.join(proj_dir, "data", "raw", subj, "**", "*.nii"), recursive=True
        )
        if len(raw_petfs) == 1:
            proc_df.at[subj, "raw_petf"] = raw_petfs[0]
        elif len(raw_petfs) == 0:
            proc_df.at[subj, "process"] = False
            proc_df.at[subj, "notes"] += "No raw PET scan found. "
        else:
            proc_df.at[subj, "process"] = False
            proc_df.at[subj, "notes"] += "Multiple raw PET scans found. "

    # Parse the raw PET scan filename to get the date and tracer.
    for subj in subjs:
        if pd.isna(proc_df.at[subj, "raw_petf"]):
            continue
        raw_petf = proc_df.at[subj, "raw_petf"]

        # Get the PET acquisition date.
        petdate_dir = raw_petf.split(op.sep)[-3]
        try:
            pet_acq_date = datetime.datetime.strptime(
                petdate_dir[:10], "%Y-%m-%d"
            ).strftime("%Y-%m-%d")
            proc_df.at[subj, "pet_date"] = pet_acq_date
        except ValueError:
            proc_df.at[subj, "process"] = False
            proc_df.at[
                subj, "notes"
            ] += "Can't parse PET acquisition date from raw PET filepath. "

        # Get the tracer.
        if np.any(
            [trac in op.basename(raw_petf).lower() for trac in ["fbb", "florbetaben"]]
        ):
            proc_df.at[subj, "tracer"] = "FBB"
        elif np.any(
            [
                trac in op.basename(raw_petf).lower()
                for trac in ["fbp", "av45", "florbetapir"]
            ]
        ):
            proc_df.at[subj, "tracer"] = "FBP"

        else:
            proc_df.at[subj, "process"] = False
            proc_df.at[subj, "notes"] += "Can't parse tracer from raw PET filename. "

        # Get the starting resolution.
        if "uniform_6mm_res" in op.basename(raw_petf).lower():
            proc_df.at[subj, "raw_res"] = [6, 6, 6]
        elif "uniform_8mm_res" in op.basename(raw_petf).lower():
            proc_df.at[subj, "raw_res"] = [8, 8, 8]
        else:
            proc_df.at[subj, "process"] = False
            proc_df.at[
                subj, "notes"
            ] += "Can't parse starting resolution from raw PET filename. "

    # Find the processed PET scan for each subject.
    for subj in subjs:
        if pd.isna(proc_df.at[subj, "raw_petf"]):
            continue
        date_tracer = "{}_{}".format(
            proc_df.at[subj, "pet_date"], proc_df.at[subj, "tracer"]
        )
        date_tracer_subj = "{}_{}".format(date_tracer, subj)
        proc_res = proc_df.at[subj, "proc_res"]
        if np.unique(proc_res).size != 1:
            proc_df.at[subj, "process"] = False
            proc_df.at[
                subj, "notes"
            ] += "Can't determine proc PET filename due to multiple unique proc_res values. "
            continue
        proc_basename = "rs{}mean_{}.nii".format(proc_res[0], date_tracer_subj)
        proc_petf = op.join(proj_dir, "data", "proc", subj, date_tracer, proc_basename)
        proc_df.at[subj, "proc_petf"] = proc_petf
        if op.isfile(proc_petf) and not overwrite:
            proc_df.at[subj, "process"] = False
            proc_df.at[subj, "processing_complete"] = True
            proc_df.at[subj, "notes"] += "Processed PET scan already exists. "

    return proc_df


def process_pet(
    petf,
    tracer,
    proj_dir,
    starting_smooth=[6, 6, 6],
    target_smooth=[8, 8, 8],
    smooth_prefix="s8",
    coreg_prefix="r",
    verbose=True,
):
    """Description: Process PET image for visual read."""
    # Smooth PET to target resolution.
    smooth_by = spm.calc_3d_smooth(starting_smooth, target_smooth)
    spetf = spm.spm_smooth(petf, smooth_by, smooth_prefix="s8")[0]

    # Coregister PET to the appropriate template.
    templatef = op.join(proj_dir, "data", "templates", "{}_template.nii".format(tracer))
    assert op.isfile(templatef)
    rspetf = spm.spm_coregister(spetf, templatef, jobtype="estwrite", out_prefix=coreg_prefix)[0]


def _parse_args():
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process ADNI amyloid PET scans for visual read."
    )
    parser.add_argument(
        "-s",
        "--subjects",
        nargs="*",
        type=str,
        help="List of subjects to process",
    )
    parser.add_argument(
        "--proj_dir",
        type=str,
        default="/Users/dschonhaut/Box/ADNI_Reads/",
        help="Path to project directory where data/, multislice/, etc. are stored",
    )
    parser.add_argument(
        "-o", "--overwrite", action="store_true", help="Overwrite existing files"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Run without printing output"
    )
    # Parse the command line arguments
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    verbose = True
    if args.quiet:
        verbose = False
    timer = Timer()

    # Get the dataframe of scans to process.
    proc_df = _get_proc_df(
        args.proj_dir, process_subjs=args.subjects, overwrite=args.overwrite
    )
    if verbose:
        print(
            "Found {}/{} subjects to process and {}/{} subjects already processed".format(
                proc_df["process"].sum(),
                len(proc_df),
                proc_df["processing_complete"].sum(),
                len(proc_df),
            )
        )

    # Process the PET image.
    for subj in proc_df.loc[proc_df["process"]].index.tolist():
        process_pet(
            petf=proc_df.at[subj, "raw_petf"],
            tracer=proc_df.at[subj, "tracer"],
            proj_dir=args.proj_dir,
            starting_smooth=proc_df.at[subj, "raw_res"],
            target_smooth=proc_df.at[subj, "proc_res"],
            smooth_prefix=proc_df.at[subj, "proc_res"][0],
        )

        if verbose:
            print(timer.loop(subj))

    if verbose:
        print(timer)
    sys.exit(0)
