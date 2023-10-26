#!/usr/bin/env python

import sys
import os
import os.path as op
import argparse
from glob import glob
import warnings
from collections import OrderedDict as od
import shutil
import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from nilearn import plotting
import general.array.array_operations as aop
from general.basic.helper_funcs import Timer
import general.basic.str_methods as strm
import general.nifti.nifti_ops as nops
import general.nifti.nifti_plotting as niiplot
import general.nifti.nifti_shell as niish
import general.nifti.spm_preproc as spm_preproc
import general.osops.os_utils as osu


mpl.rcParams["pdf.fonttype"] = 42


def find_scans_to_process(
    base_dir,
    raw_dirname="raw",
    proc_dirname="proc",
    process_subjs=None,
    proc_res=[8, 8, 8],
    skip_smooth=False,
    skip_coreg=True,
    overwrite=False,
    save_output=True,
    verbose=True,
):
    """Get a dataframe of subjects to process.

    Parameters
    ----------
    base_dir : str
        Path to the base directory of the project.
    raw_dirname : str
        Name of the raw data directory.
    proc_dirname : str
        Name of the processed data directory.
    process_subjs : list of str
        List of subject IDs to process. If None, all subjects will be
        processed.
    proc_res : list[int, int, int]
        Final resolution of the processed PET scans.
    skip_smooth : bool
        If True, skip smoothing the PET scans.
    skip_coreg : bool
        If True, skip coregistering the PET scans to the target image.
    overwrite : bool
        If True, overwrite existing processed PET scans.

    Returns
    -------
    pet_proc : pandas.DataFrame
        A dataframe of subjects to process.
    """
    # Get a list of subjects in the raw directory.
    subjs = []
    for d in os.listdir(op.join(base_dir, "data", raw_dirname)):
        if d.startswith("sub-"):
            subjs.append(d.split("-")[1])
        else:
            subjs.append(d)

    # Initialize the output dataframe.
    pet_proc = pd.DataFrame(
        index=subjs,
        columns=[
            "pet_date",
            "tracer",
            "raw_petf",
            "raw_cp_petf",
            "proc_petf",
            "multislicef",
            "merged_multislicef",
            "input_res",
            "proc_res",
            "skip_smooth",
            "skip_coreg",
            "already_processed",
            "to_process",
            "just_processed",
            "notes",
        ],
    )
    pet_proc["raw_petf"] = ""
    pet_proc["raw_cp_petf"] = ""
    pet_proc["proc_res"] = [proc_res] * len(pet_proc)
    pet_proc["already_processed"] = False
    pet_proc["to_process"] = True
    pet_proc["just_processed"] = False
    pet_proc["notes"] = ""
    if process_subjs:
        _process_subjs = []
        for subj in process_subjs:
            if subj.startswith("sub-"):
                _process_subjs.append(subj.split("-")[1])
            else:
                _process_subjs.append(subj)
        exclude_subjs = [subj for subj in pet_proc.index if subj not in _process_subjs]
        pet_proc.loc[exclude_subjs, "to_process"] = False

    # Find the raw PET scan for each subject.
    for subj in subjs:
        raw_pet_files = _get_raw_files(base_dir, raw_dirname, subj)
        if len(raw_pet_files) == 1:
            pet_proc.at[subj, "raw_petf"] = raw_pet_files[0]
        elif len(raw_pet_files) == 0:
            pet_proc.at[subj, "to_process"] = False
            pet_proc.at[subj, "notes"] += "No raw PET scan found. "
        else:
            pet_proc.at[subj, "to_process"] = False
            pet_proc.at[subj, "notes"] += "Multiple raw PET scans found. "

    # Parse the raw PET filepath to determine PET acquisition date,
    # tracer, and starting resolution.
    for subj in subjs:
        if pet_proc.at[subj, "to_process"] is False:
            continue

        raw_petf = pet_proc.at[subj, "raw_petf"]

        # Get the PET acquisition date.
        # print(subj, pet_proc.loc[subj])
        pet_date_dir = raw_petf.split(op.sep)[-3]
        try:
            pet_date = datetime.datetime.strptime(
                pet_date_dir[:10], "%Y-%m-%d"
            ).strftime("%Y-%m-%d")
            pet_proc.at[subj, "pet_date"] = pet_date
        except ValueError:
            pet_proc.at[subj, "to_process"] = False
            pet_proc.at[
                subj, "notes"
            ] += "Can't parse PET acquisition date from the raw PET filepath. "

        # Get the tracer.
        tracer = _get_tracer(op.basename(raw_petf))
        if tracer is not None:
            pet_proc.at[subj, "tracer"] = tracer
        else:
            pet_proc.at[subj, "to_process"] = False
            pet_proc.at[subj, "notes"] += "Can't parse tracer from raw PET filename. "

        # Get the starting resolution.
        if "uniform_6mm_res" in op.basename(raw_petf).lower():
            pet_proc.at[subj, "input_res"] = [6, 6, 6]
        elif "uniform_8mm_res" in op.basename(raw_petf).lower():
            pet_proc.at[subj, "input_res"] = [8, 8, 8]
        elif ("uniform_" in op.basename(raw_petf).lower()) and (
            "mm_res" in op.basename(raw_petf).lower()
        ):
            # Find the substring in between "uniform_" and "mm_res".
            input_res = float(
                op.basename(raw_petf).lower().split("uniform_")[1].split("mm_res")[0]
            )
            pet_proc.at[subj, "input_res"] = [input_res, input_res, input_res]
        else:
            pet_proc.at[subj, "to_process"] = False
            pet_proc.at[
                subj, "notes"
            ] += "Can't parse starting resolution from raw PET filename. "

    # Find the processed PET scan for each subject.
    for subj in subjs:
        if pet_proc.at[subj, "to_process"] is False:
            continue
        elif np.unique(pet_proc.at[subj, "proc_res"]).size != 1:
            pet_proc.at[subj, "to_process"] = False
            pet_proc.at[
                subj, "notes"
            ] += "Can't determine proc PET filename due to multiple unique proc_res values. "
            continue
        elif np.any(
            np.asanyarray(pet_proc.at[subj, "proc_res"])
            < np.asanyarray(pet_proc.at[subj, "input_res"])
        ):
            pet_proc.at[subj, "to_process"] = False
            pet_proc.at[
                subj, "notes"
            ] += "Can't smooth PET to a lower final resolution than input resolution. "
            continue

        # Figure out what processing steps will be performed.
        if skip_smooth or np.array_equal(
            pet_proc.at[subj, "input_res"], pet_proc.at[subj, "proc_res"]
        ):
            pet_proc.at[subj, "skip_smooth"] = True
        else:
            pet_proc.at[subj, "skip_smooth"] = False
        pet_proc.at[subj, "skip_coreg"] = skip_coreg

        # Get the processed PET filepaths.
        proc_pet_files = _get_proc_files(
            base_dir=base_dir,
            proc_dirname=proc_dirname,
            subj=subj,
            tracer=pet_proc.at[subj, "tracer"],
            pet_date=pet_proc.at[subj, "pet_date"],
            proc_res=pet_proc.at[subj, "proc_res"],
            skip_smooth=pet_proc.at[subj, "skip_smooth"],
            skip_coreg=pet_proc.at[subj, "skip_coreg"],
        )
        for key, val in proc_pet_files.items():
            pet_proc.at[subj, key] = val
        check_files = ["proc_petf", "multislicef", "merged_multislicef"]
        if np.all([op.isfile(proc_pet_files[key]) for key in check_files]):
            pet_proc.at[subj, "already_processed"] = True
            if not overwrite:
                pet_proc.at[subj, "to_process"] = False
        for key in check_files:
            if not op.isfile(pet_proc.at[subj, key]):
                pet_proc.at[subj, key] = ""

    return pet_proc


def _get_raw_files(base_dir, raw_dirname, subj):
    raw_pet_files = glob(
        op.join(base_dir, "data", raw_dirname, subj, "**", "*.nii*"),
        recursive=True,
    )
    return raw_pet_files


def _get_proc_files(
    base_dir, proc_dirname, subj, tracer, pet_date, proc_res, skip_smooth, skip_coreg
):
    proc_dir = op.join(
        base_dir,
        "data",
        proc_dirname,
        subj,
        f"pet-{tracer}",
        f"ses-{pet_date}",
    )
    intermed_dir = op.join(proc_dir, "intermed")
    proc_files = od([])
    proc_files["raw_cp_petf"] = op.join(
        intermed_dir, f"mean_{subj}_pet-{tracer}_ses-{pet_date}.nii"
    )
    if skip_smooth and skip_coreg:
        proc_files["proc_petf"] = op.join(
            proc_dir, op.basename(proc_files["raw_cp_petf"])
        )
    elif skip_coreg and not skip_smooth:
        proc_files["proc_petf"] = op.join(
            proc_dir,
            strm.add_presuf(
                op.basename(proc_files["raw_cp_petf"]), prefix=f"s{proc_res[0]}"
            ),
        )
    elif skip_smooth and not skip_coreg:
        proc_files["proc_petf"] = op.join(
            proc_dir,
            strm.add_presuf(op.basename(proc_files["raw_cp_petf"]), prefix="r"),
        )
    else:
        proc_files["proc_petf"] = op.join(
            proc_dir,
            strm.add_presuf(
                op.basename(proc_files["raw_cp_petf"]), prefix=f"rs{proc_res[0]}"
            ),
        )
    for key in proc_files:
        proc_files[key] = nops.find_gzip(proc_files[key], return_infile=True)
    proc_files["multislicef"] = op.join(
        proc_dir,
        strm.add_presuf(op.basename(proc_files["proc_petf"]), suffix="_multislice")
        .replace(".nii.gz", ".pdf")
        .replace(".nii", ".pdf"),
    )
    proc_files["merged_multislicef"] = strm.add_presuf(
        proc_files["multislicef"], suffix="_merged"
    )
    return proc_files


def _get_tracer(basename):
    """Return the PET tracer used from the file basename.

    Tracers searched
    ----------------
    Florbetaben
    Florbetapir
    Flutafuranol
    PIB

    Parameters
    ----------
    basename : str
        The basename of the input file.

    Returns
    -------
    tracer : str
        The PET tracer parsed from the input file basename.
    """
    if np.any(
        [tracer in basename.lower() for tracer in ["fbb", "florbetaben", "neuraceq"]]
    ):
        tracer = "FBB"
        return tracer
    elif np.any(
        [
            tracer in basename.lower()
            for tracer in ["fbp", "florbetapir", "av45", "av-45", "amyvid"]
        ]
    ):
        tracer = "FBP"
        return tracer
    elif np.any(
        [
            tracer in basename.lower()
            for tracer in ["flutafuranol", "nav4694", "nav-4694", "azd4694", "azd-4694"]
        ]
    ):
        tracer = "NAV"
        return tracer
    elif np.any(
        [
            tracer in basename.lower()
            for tracer in ["pib", "pittsburgh compound b", "pittsburgh compound-b"]
        ]
    ):
        tracer = "PIB"
        return tracer
    else:
        return None


def _save_logfile(base_dir, pet_proc, verbose=True):
    """Save the processing dataframe to a csv file."""
    # Get the current date and time in string format.
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outfile = op.join(base_dir, "logs", f"pet_proc_{now}.csv")
    try:
        if op.exists(outfile):
            os.remove(outfile)
        pet_proc.to_csv(outfile, index=True, index_label="subj")
        if verbose:
            print(f"\nSaved logfile: {outfile}")
        return outfile
    except (PermissionError, BlockingIOError):
        print(f"Could not save logfile {outfile}")


def process_pet(
    base_dir,
    raw_petf,
    raw_cp_petf,
    tracer,
    input_res=[6, 6, 6],
    proc_res=[8, 8, 8],
    coreg_dof=6,
    skip_smooth=False,
    skip_coreg=True,
    use_spm=True,
    gzip_niftis=True,
    verbose=True,
):
    """Process PET image for visual read.

    Returns
    -------
    outfiles : list
        List of files generated during processing (in order from
        first to last). Copied files are included, but initial names for
        renamed files are not retained. At time of return all outfiles
        therefore exist.
    """
    if verbose:
        subj = raw_cp_petf.split(op.sep)[-5].split("-")[1]
        print("\n{}\n{}".format(subj, "-" * len(subj)))

    # Delete existing processed files.
    proc_dir = op.dirname(op.dirname(raw_cp_petf))
    osu.rm_files(proc_dir)

    # Setup proc directory structure.
    if verbose:
        print("  Setting up directories...")
    os.makedirs(op.dirname(raw_cp_petf), exist_ok=True)
    os.chdir(op.dirname(raw_cp_petf))
    outfiles = []

    # Copy out the raw PET file.
    if verbose:
        print(
            "  Copying PET from {}/ to {}/...".format(
                raw_petf.replace(base_dir, "").split("/")[2],
                raw_cp_petf.replace(base_dir, "").split("/")[2],
            )
        )
    # Make sure the copy file has the correct extension.
    if raw_petf.endswith(".nii.gz") and raw_cp_petf.endswith(".nii"):
        raw_cp_petf += ".gz"
    elif raw_petf.endswith(".nii") and raw_cp_petf.endswith(".nii.gz"):
        raw_cp_petf = raw_cp_petf[:-3]
    _outfile = shutil.copy(src=raw_petf, dst=raw_cp_petf)
    outfiles.append(_outfile)

    # Reset origin to center.
    if verbose:
        print("  Resetting origin to center...")
    *_, _outfile = nops.recenter_nii(outfiles[-1], verbose=False)
    if _outfile != outfiles[-1]:
        outfiles.append(_outfile)

    # Smooth PET to target resolution.
    if not skip_smooth:
        if use_spm:
            if verbose:
                print("  Smoothing PET (SPM)...")
            if outfiles[-1].endswith(".nii.gz"):
                outfiles[-1] = nops.gunzip_nii(outfiles[-1])
            _outfile = spm_preproc.spm_smooth(
                outfiles[-1],
                res_in=input_res,
                res_target=proc_res,
                prefix=f"s{proc_res[0]}",
            )[0]
        else:
            if verbose:
                print("  Smoothing PET (niimath)...")
            _outfile = niish.niimath_smooth(
                outfiles[-1],
                res_in=input_res,
                res_target=proc_res,
                prefix=f"s{proc_res[0]}",
                verbose=False,
            )
        if _outfile != outfiles[-1]:
            outfiles.append(_outfile)

    # Coregister PET to standard space.
    if not skip_coreg:
        targetf = nops.find_gzip(
            op.join(base_dir, "templates", f"rTemplate_{tracer}-all.nii"),
            raise_error=True,
        )
        if use_spm:
            if verbose:
                print(
                    f"  Applying 6-degree linear coreg and reslicing PET to target res (SPM12)..."
                )
            if outfiles[-1].endswith(".nii.gz"):
                outfiles[-1] = nops.gunzip_nii(outfiles[-1])
            if targetf.endswith(".nii.gz"):
                targetf = nops.gunzip_nii(targetf)
            jobtype = "estwrite"
            _outfile = spm_preproc.spm_coregister(
                source=outfiles[-1], target=targetf, jobtype=jobtype, out_prefix="r"
            )[0]
        else:
            if verbose:
                print(
                    f"  Applying {coreg_dof}-degree linear coreg and reslicing PET to target res (FSL)..."
                )
            _outfile = niish.fsl_flirt(
                infile=outfiles[-1],
                target=targetf,
                dof=coreg_dof,
                prefix="r",
                verbose=False,
            )
        if _outfile != outfiles[-1]:
            outfiles.append(_outfile)

    # Move the final PET image to its parent directory.
    _outfile = op.join(op.dirname(op.dirname(outfiles[-1])), op.basename(outfiles[-1]))
    _outfile = shutil.move(src=outfiles[-1], dst=_outfile)
    outfiles[-1] = _outfile
    if verbose:
        print(
            "  Done!\n"
            + "  Processed PET image: {}/\n".format(op.dirname(outfiles[-1]))
            + "                       {}".format(op.basename(outfiles[-1]))
        )

    # Gzip or gunzip the processed and intermediary niftis.
    for ii in range(len(outfiles)):
        if gzip_niftis:
            outfiles[ii] = nops.gzip_nii(outfiles[ii])
        else:
            outfiles[ii] = nops.gunzip_nii(outfiles[ii])

    return outfiles


def merge_multislice(
    infile,
    template_dir,
    tracer,
    remove_infile=False,
    overwrite=False,
    verbose=True,
):
    """Merge multislice PDF of a scan with the matching tracer template.

    Parameters
    ----------
    infile : str
        The multi-slice PDF to merge with the template.
    tracer : str
        The PET tracer.
    remove_infile : bool
        Whether to remove the input multislice file.
    overwrite : bool
        Whether to overwrite the output file if it already exists.
    verbose : bool
        Whether to print status messages.

    Returns
    -------
    outfile : str
        The merged PDF.
    """
    assert infile.endswith(".pdf")

    # Return the outfile if it already exists.
    outfile = strm.add_presuf(infile, suffix="_merged")
    if op.isfile(outfile) and not overwrite:
        if verbose:
            print(
                "  Merged PDF: {}\n".format(op.dirname(outfile))
                + "              {}".format(op.basename(outfile))
            )
        return outfile

    # Load the template PDF
    templatef = op.join(template_dir, f"ADNI4_{tracer}_template.pdf")

    # Merge the PDFs.
    cmd = f"qpdf --linearize --qdf --optimize-images --empty --pages {templatef} 1 {infile} 1 -- {outfile}"
    _ = osu.run_cmd(cmd)
    if verbose:
        print(
            "  Merged PDF: {}\n".format(op.dirname(outfile))
            + "              {}".format(op.basename(outfile))
        )

    # Remove the intermediary file.
    if remove_infile:
        os.remove(infile)
        if verbose:
            print(f"  Removed {infile}")

    return outfile


def _parse_args():
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(
        description="""Process ADNI amyloid PET scans for visual read.

steps:
  [1] Copy PET scan from [base_dir]/data/[raw_dirname]/[subject]/[nested_dirs_from_LONI]/[pet_scan].nii
      to [base_dir]/data/[proc_dirname]/[subject]/pet-[tracer]/ses-[pet_date]/intermed/
      mean_[subject]_pet-[tracer]_ses-[pet_date].nii
  [2] Reset origin to center (saves over header info of the copied image)
  [3] Smooth PET to a defined resolution (default 8mm isotropic). Note that if the input
      and final resolutions are the same, smoothing is skipped (optional step,
      default=False)
  [4] [Coregister and reslice PET (default 6-degree rigid body transform) to a standard
       space (optional step, default=False)]
  [5] Gzip NIfTI files (optional step, default=False)
  [6] Save a PDF of axial multislices of the processed PET scan and a merged PDF
      that also shows canonical positive and negative scans for the same tracer""",
        formatter_class=TextFormatter,
        exit_on_error=False,
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/Volumes/petcore/Projects/ADNI_Reads",
        help=(
            "Path to base directory where data/ and templates/ are stored\n"
            + "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--raw_dirname",
        type=str,
        default="raw",
        help=(
            "Name of the subdirectory in [base_dir]/data where raw PET files for scans\n"
            + "to process are stored (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--proc_dirname",
        type=str,
        default="proc",
        help=(
            "Name of the subdirectory in [base_dir]/data where processed PET and\n"
            + "multislice files are stored (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "-s",
        "--subjects",
        nargs="*",
        type=str,
        help=(
            "List of subjects to process. If --subjects is not defined and\n"
            + "--overwrite is not defined, then all unprocessed subjects in\n"
            + "[base_dir]/data/[raw_dirname]/ will be processed. If --overwrite is\n"
            + "defined, then all subjects in [base_dir]/data/raw/ will be processed"
        ),
    )
    parser.add_argument(
        "--smooth",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Smooth PET to the resolution defined by --final_res",
    )
    parser.add_argument(
        "--final_res",
        default=8,
        type=float,
        help="Final resolution (FWHM) of smoothed PET, in mm (default: %(default)s)",
    )
    parser.add_argument(
        "--coreg",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Coregister and reslice PET to standard space",
    )
    parser.add_argument(
        "--coreg_dof",
        type=int,
        default=6,
        choices=[3, 6, 9, 12],
        help=(
            "Degrees of freedom used for linear coregistration\n"
            + " 3 = translation only\n"
            + " 6 = translation + rotation (rigid body)\n"
            + " 9 = translation + rotation + scaling\n"
            + "12 = translation + rotation + scaling + shearing (full affine)\n"
            + "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--use_spm",
        action="store_true",
        help="Use SPM for processing, instead of default FSL and niimath",
    )
    parser.add_argument(
        "--gzip",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Gzip processed NIfTI files (raw files are untouched)",
    )
    parser.add_argument(
        "--skip_proc",
        action="store_true",
        help=(
            "Skip PET processing and jump straight to multislice PDF creation.\n"
            + "Requires PET to have already been processed, otherwise PET\n"
            + "processing is still complated. Use this flag with --overwrite to\n"
            + "keep PET processing untouched but overwrite multislice PDFs"
        ),
    )
    parser.add_argument(
        "--skip_multislice",
        action="store_true",
        help="Process PET but skip multislice PDF creation",
    )
    parser.add_argument(
        "-z",
        "--slices",
        type=int,
        nargs="+",
        default=[-50, -37, -24, -11, 2, 15, 28, 41],
        # [-32, -18, -4, 10, 24, 38, 52] [-51, -30, -21, -6, 9, 24, 39, 54]
        help=(
            "List of image slices to show along the z-axis, in MNI coordinates\n"
            + "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--crop",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Crop the multislice images to the brain",
    )
    parser.add_argument(
        "--crop_cut",
        type=float,
        default=0.05,
        help=(
            "Defines how tightly to crop the brain for multislice creation\n"
            + "(proportion of voxels > 0 in each plane that are allowed to be cropped)\n"
            + "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--cmap",
        type=str,
        help=(
            "Colormap to use for the multislice images (overrides the\n"
            + "tracer-specific defaults)"
        ),
    )
    parser.add_argument(
        "--autoscale",
        action="store_true",
        help=(
            "Set multislice vmin and vmax to to 0.01 and the 99.5th percentile\n"
            + "of nonzero values, respectively (overrides --vmin, --vmax,\n"
            + "and tracer-specific default scaling)"
        ),
    )
    parser.add_argument(
        "--vmin",
        type=float,
        help=(
            "Minimum intensity threshold for the multislice images\n"
            + "(overrides the tracer-specific defaults)"
        ),
    )
    parser.add_argument(
        "--vmax",
        type=float,
        help=(
            "Maximum intensity threshold for the multislice images\n"
            + "(overrides the tracer-specific defaults)"
        ),
    )
    parser.add_argument(
        "-o", "--overwrite", action="store_true", help="Overwrite existing files"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Run without printing output"
    )

    # Parse the command line arguments
    args = parser.parse_args()
    if (len(sys.argv) == 1) and not op.isdir(args.base_dir):
        parser.print_help()
        sys.exit()
    return args


def create_read_file():
    """Create a visual read CSV file for the reader to complete."""
    pass


class TextFormatter(argparse.RawTextHelpFormatter):
    """Custom formatter for argparse help text."""

    # use defined argument order to display usage
    def _format_usage(self, usage, actions, groups, prefix):
        if prefix is None:
            prefix = "usage: "

        # if usage is specified, use that
        if usage is not None:
            usage = usage % dict(prog=self._prog)

        # if no optionals or positionals are available, usage is just prog
        elif usage is None and not actions:
            usage = "%(prog)s" % dict(prog=self._prog)
        elif usage is None:
            prog = "%(prog)s" % dict(prog=self._prog)
            # build full usage string
            action_usage = self._format_actions_usage(actions, groups)  # NEW
            usage = " ".join([s for s in [prog, action_usage] if s])
            # omit the long line wrapping code
        # prefix with 'usage:'
        return "%s%s\n\n" % (prefix, usage)


if __name__ == "__main__":
    # Start the timer.
    timer = Timer(msg="\nTotal runtime: ")

    # Get command line arguments.
    args = _parse_args()

    # Format arguments.
    proc_res = [args.final_res] * 3
    skip_smooth = True
    if args.smooth:
        skip_smooth = False
    skip_coreg = True
    if args.coreg:
        skip_coreg = False
    use_spm = False
    if args.use_spm:
        use_spm = True
        if args.coreg_dof != 6:
            print("SPM only supports 6-degree coregn; overriding --coreg_dof to 6")
            args.coreg_dof = 6
    verbose = True
    if args.quiet:
        verbose = False

    # Get the dataframe of scans to process.
    pet_proc = find_scans_to_process(
        base_dir=args.base_dir,
        raw_dirname=args.raw_dirname,
        proc_dirname=args.proc_dirname,
        process_subjs=args.subjects,
        proc_res=proc_res,
        skip_smooth=skip_smooth,
        skip_coreg=skip_coreg,
        overwrite=args.overwrite,
    )
    if verbose:
        print(
            "{}/{} subjects in {} have already been processed in data/{}/".format(
                pet_proc["already_processed"].sum(),
                len(pet_proc),
                op.join(args.base_dir, "data", args.raw_dirname),
                args.proc_dirname,
            )
        )
        print(
            "{}/{} subjects will now be processed".format(
                pet_proc["to_process"].sum(),
                len(pet_proc),
            )
        )

    # Process each scan and save a multislice PDF of the processed
    # image.
    for subj in pet_proc.query("to_process == True").index:
        if op.isfile(pet_proc.at[subj, "proc_petf"]) and (
            args.skip_proc or not args.overwrite
        ):
            if verbose:
                subj = pet_proc.at[subj, "raw_cp_petf"].split(op.sep)[-5].split("-")[1]
                print("\n{}\n{}".format(subj, "-" * len(subj)))
                print("  Skipping PET processing (already complete)...")
                print(
                    "  Processed PET image: {}/\n".format(
                        op.dirname(pet_proc.at[subj, "proc_petf"])
                    )
                    + "                       {}".format(
                        op.basename(pet_proc.at[subj, "proc_petf"])
                    )
                )
        else:
            # Process the PET image.
            outfiles = process_pet(
                base_dir=args.base_dir,
                raw_petf=pet_proc.at[subj, "raw_petf"],
                raw_cp_petf=pet_proc.at[subj, "raw_cp_petf"],
                tracer=pet_proc.at[subj, "tracer"],
                input_res=pet_proc.at[subj, "input_res"],
                proc_res=pet_proc.at[subj, "proc_res"],
                coreg_dof=args.coreg_dof,
                skip_smooth=skip_smooth,
                skip_coreg=skip_coreg,
                use_spm=use_spm,
                gzip_niftis=args.gzip,
                verbose=verbose,
            )
            pet_proc.at[subj, "raw_cp_petf"] = outfiles[0]
            pet_proc.at[subj, "proc_petf"] = outfiles[-1]

        if args.skip_multislice:
            if verbose:
                print("  Skipping multislice creation...")
        else:
            # Create the multislice PDF.
            multislicef = niiplot.create_multislice(
                imagef=pet_proc.at[subj, "proc_petf"],
                subj=subj,
                tracer=pet_proc.at[subj, "tracer"],
                image_date=pet_proc.at[subj, "pet_date"],
                cut_coords=args.slices,
                cmap=args.cmap,
                vmin=args.vmin,
                vmax=args.vmax,
                autoscale=args.autoscale,
                crop=args.crop,
                crop_cut=args.crop_cut,
                overwrite=args.overwrite,
                verbose=verbose,
            )
            pet_proc.at[subj, "multislicef"] = multislicef

            # Merge the subject's multislice images with the tracer
            # template.
            merged_multislicef = merge_multislice(
                infile=multislicef,
                template_dir=op.join(args.base_dir, "templates"),
                tracer=pet_proc.at[subj, "tracer"],
                remove_infile=False,
                overwrite=args.overwrite,
                verbose=verbose,
            )
            pet_proc.at[subj, "merged_multislicef"] = merged_multislicef

        # Create a visual read file for this subject.
        pass

        # Print the runtime for this subject.
        pet_proc.at[subj, "just_processed"] = True
        if verbose:
            timer.loop(f"  Runtime")

    # Save the logfile.
    logfile = _save_logfile(args.base_dir, pet_proc, verbose)

    # Print the total runtime.
    if verbose:
        print(timer)

    sys.exit(0)
