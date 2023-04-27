#!/Users/dschonhaut/mambaforge/envs/nipy310/bin/python


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
import matplotlib
import matplotlib.pyplot as plt
from nilearn import plotting
import PyPDF2
import pikepdf

sys.path.append(op.join(op.expanduser("~"), "code"))
import general.nifti.nifti_ops as nops
import general.nifti.spm_preproc as spm_preproc
import general.nifti.nifti_shell as niish
from general.basic.helper_funcs import Timer
import general.basic.str_methods as strm
import general.osops.os_utils as osu

matplotlib.rcParams["pdf.fonttype"] = 42


def find_scans_to_process(
    base_dir,
    process_subjs=None,
    proc_res=[8, 8, 8],
    overwrite=False,
    save_output=True,
    verbose=True,
):
    """Get a dataframe of subjects to process.

    Parameters
    ----------
    base_dir : str
        Path to the base directory of the project.
    process_subjs : list of str
        List of subject IDs to process. If None, all subjects will be
        processed.
    proc_res : list[int, int, int]
        Final resolution of the processed PET scans.
    overwrite : bool
        If True, overwrite existing processed PET scans.
    save_output : bool
        If True, save the output dataframe to a csv file in
        base_dir/data.
    verbose : bool
        If True, print status messages.

    Returns
    -------
    proc_df : pandas.DataFrame
        A dataframe of subjects to process.
    """
    # Get a list of subjects in the raw directory.
    subjs = [d.split("-")[1] for d in os.listdir(op.join(base_dir, "data", "raw"))]

    # Initialize the output dataframe.
    proc_df = pd.DataFrame(
        index=subjs,
        columns=[
            "pet_date",
            "tracer",
            "raw_petf",
            "mean_petf",
            "smean_petf",
            "rsmean_petf",
            "multislicef",
            "input_res",
            "proc_res",
            "process",
            "processing_complete",
            "notes",
        ],
    )
    proc_df["proc_res"] = [proc_res] * len(proc_df)
    proc_df["process"] = True
    proc_df["processing_complete"] = False
    proc_df["notes"] = ""
    if process_subjs:
        exclude_subjs = [subj for subj in proc_df.index if subj not in process_subjs]
        proc_df.loc[exclude_subjs, "process"] = False

    # Find the raw PET scan for each subject.
    for subj in subjs:
        raw_pet_files = _get_raw_files(base_dir, subj)
        if len(raw_pet_files) == 1:
            proc_df.at[subj, "raw_petf"] = raw_pet_files[0]
        elif len(raw_pet_files) == 0:
            proc_df.at[subj, "process"] = False
            proc_df.at[subj, "notes"] += "No raw PET scan found. "
        else:
            proc_df.at[subj, "process"] = False
            proc_df.at[subj, "notes"] += "Multiple raw PET scans found. "

    # Parse the raw PET filepath to determine PET acquisition date,
    # tracer, and starting resolution.
    for subj in subjs:
        if proc_df.at[subj, "process"] is False:
            continue

        raw_petf = proc_df.at[subj, "raw_petf"]

        # Get the PET acquisition date.
        pet_date_dir = raw_petf.split(op.sep)[-3]
        try:
            pet_date = datetime.datetime.strptime(
                pet_date_dir[:10], "%Y-%m-%d"
            ).strftime("%Y-%m-%d")
            proc_df.at[subj, "pet_date"] = pet_date
        except ValueError:
            proc_df.at[subj, "process"] = False
            proc_df.at[
                subj, "notes"
            ] += "Can't parse PET acquisition date from the raw PET filepath. "

        # Get the tracer.
        tracer = _get_tracer(op.basename(raw_petf))
        if tracer is not None:
            proc_df.at[subj, "tracer"] = tracer
        else:
            proc_df.at[subj, "process"] = False
            proc_df.at[subj, "notes"] += "Can't parse tracer from raw PET filename. "

        # Get the starting resolution.
        if "uniform_6mm_res" in op.basename(raw_petf).lower():
            proc_df.at[subj, "input_res"] = [6, 6, 6]
        elif "uniform_8mm_res" in op.basename(raw_petf).lower():
            proc_df.at[subj, "input_res"] = [8, 8, 8]
        elif ("uniform_" in op.basename(raw_petf).lower()) and (
            "mm_res" in op.basename(raw_petf).lower()
        ):
            # Find the substring in between "uniform_" and "mm_res".
            input_res = float(
                op.basename(raw_petf).lower().split("uniform_")[1].split("mm_res")[0]
            )
            proc_df.at[subj, "input_res"] = [input_res, input_res, input_res]
        else:
            proc_df.at[subj, "process"] = False
            proc_df.at[
                subj, "notes"
            ] += "Can't parse starting resolution from raw PET filename. "

    # Find the processed PET scan for each subject.
    for subj in subjs:
        if proc_df.at[subj, "process"] is False:
            continue
        elif np.unique(proc_df.at[subj, "proc_res"]).size != 1:
            proc_df.at[subj, "process"] = False
            proc_df.at[
                subj, "notes"
            ] += "Can't determine proc PET filename due to multiple unique proc_res values. "
            continue
        elif np.any(
            np.asanyarray(proc_df.at[subj, "proc_res"])
            < np.asanyarray(proc_df.at[subj, "input_res"])
        ):
            proc_df.at[subj, "process"] = False
            proc_df.at[
                subj, "notes"
            ] += "Can't smooth PET to a lower final resolution than input resolution. "
            continue

        proc_pet_files = _get_proc_files(
            base_dir=base_dir,
            subj=subj,
            tracer=proc_df.at[subj, "tracer"],
            pet_date=proc_df.at[subj, "pet_date"],
            proc_res=proc_df.at[subj, "proc_res"],
        )
        for key, val in proc_pet_files.items():
            proc_df.at[subj, key] = val
        if np.all(
            [
                op.isfile(nops.find_gzip(f, return_infile=True))
                for f in proc_pet_files.values()
            ]
        ):
            proc_df.at[subj, "processing_complete"] = True
            if not overwrite:
                proc_df.at[subj, "process"] = False

    # Save the output dataframe to a csv file.
    if save_output:
        outfile = op.join(base_dir, "data", ".proc_df.csv")
        try:
            if op.exists(outfile):
                os.remove(outfile)
            proc_df.to_csv(outfile, index=True, index_label="subj")
        except (PermissionError, BlockingIOError):
            print(f"Could not save {outfile}")
        if verbose:
            print(f"Saved {outfile}")

    return proc_df


def _get_raw_files(base_dir, subj):
    raw_pet_files = glob(
        op.join(base_dir, "data", "raw", f"sub-{subj}", "**", "*.nii*"), recursive=True
    )
    return raw_pet_files


def _get_proc_files(base_dir, subj, tracer, pet_date, proc_res):
    proc_dir = op.join(
        base_dir, "data", "proc", f"sub-{subj}", f"pet-{tracer}", f"ses-{pet_date}"
    )
    intermed_dir = op.join(proc_dir, "intermed")
    proc_files = od(
        [
            (
                "mean_petf",
                op.join(
                    intermed_dir, f"mean_sub-{subj}_pet-{tracer}_ses-{pet_date}.nii"
                ),
            ),
            (
                "smean_petf",
                op.join(
                    intermed_dir,
                    f"s{proc_res[0]}mean_sub-{subj}_pet-{tracer}_ses-{pet_date}.nii",
                ),
            ),
            (
                "rsmean_petf",
                op.join(
                    proc_dir,
                    f"rs{proc_res[0]}mean_sub-{subj}_pet-{tracer}_ses-{pet_date}.nii",
                ),
            ),
            (
                "multislicef",
                op.join(
                    proc_dir,
                    f"rs{proc_res[0]}mean_sub-{subj}_pet-{tracer}_ses-{pet_date}_multislice.pdf",
                ),
            ),
        ]
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


def process_pet(
    base_dir,
    raw_petf,
    mean_petf,
    tracer,
    input_res=[6, 6, 6],
    proc_res=[8, 8, 8],
    coreg_dof=6,
    use_spm=True,
    verbose=True,
):
    """Process PET image for visual read."""
    if verbose:
        subj = mean_petf.split(op.sep)[-5].split("-")[1]
        print("\n{}\n{}".format(subj, "-" * len(subj)))

    # Setup proc directory structure.
    if verbose:
        print("  Setting up proc/ directory structure")
    os.makedirs(op.dirname(mean_petf), exist_ok=True)
    os.chdir(op.dirname(mean_petf))

    # Copy out the raw PET file.
    if verbose:
        print("  Copying mean PET from raw/ to proc/")
    _ = shutil.copy(src=raw_petf, dst=mean_petf)

    # Reset origin to center.
    if verbose:
        print("  Resetting origin to center")
    _ = nops.recenter_nii(mean_petf, verbose=False)

    # Smooth PET to target resolution.
    if np.all(np.asanyarray(input_res) == np.asanyarray(proc_res)):
        smean_petf = strm.add_presuf(mean_petf, prefix=f"s{proc_res[0]}")
        _ = shutil.copy(src=mean_petf, dst=smean_petf)
    else:
        if use_spm:
            if verbose:
                print("  Smoothing PET (SPM)")
            outfiles_smooth = spm_preproc.spm_smooth(
                mean_petf,
                res_in=input_res,
                res_target=proc_res,
                prefix=f"s{proc_res[0]}",
            )
            assert len(outfiles_smooth) == 1
            smean_petf = outfiles_smooth[0]
        else:
            if verbose:
                print("  Smoothing PET (niimath)")
            smean_petf = niish.niimath_smooth(
                mean_petf,
                res_in=input_res,
                res_target=proc_res,
                prefix=f"s{proc_res[0]}",
                verbose=verbose,
            )

    # Coregister PET to standard space.
    templatef = op.join(base_dir, "templates", f"rTemplate_{tracer}-all.nii")
    if use_spm:
        if verbose:
            print(
                f"  Applying 6-degree linear coregistration and reslicing PET to standard space (SPM12)"
            )
        jobtype = "estwrite"
        outfiles_coreg = spm_preproc.spm_coregister(
            source=smean_petf, target=templatef, jobtype=jobtype, out_prefix="r"
        )
        assert len(outfiles_coreg) == 1
        _rsmean_petf = outfiles_coreg[0]
    else:
        if verbose:
            print(
                f"  Applying {coreg_dof}-degree linear coregistration and reslicing PET to standard space (FSL)"
            )
        _rsmean_petf = niish.fsl_flirt(
            infile=smean_petf,
            target=templatef,
            dof=coreg_dof,
            prefix="r",
            verbose=verbose,
        )

    # Move the coregistered PET file from intermed/ to its parent
    # directory.
    if verbose:
        print("  Moving coregistered PET from intermed/ to parent directory")
    rsmean_petf = op.join(
        op.dirname(op.dirname(_rsmean_petf)), op.basename(_rsmean_petf)
    )
    _ = shutil.move(src=_rsmean_petf, dst=rsmean_petf)

    if verbose:
        print("  PET processing done\n")

    return rsmean_petf


def create_multislice(
    imagef,
    subj,
    tracer,
    pet_date,
    display_mode="z",
    cut_coords=[-51, -30, -21, -6, 9, 24, 39, 54],
    annotate=False,
    draw_cross=False,
    colorbar=False,
    cbar_tick_format="%.2f",
    figsize=(13.33, 7.5),
    dpi=300,
    font={"tick": 12, "label": 14, "title": 20, "annot": 12},
    cmap=None,
    vmin=None,
    vmax=None,
    title=None,
    fig=None,
    ax=None,
    overwrite=False,
    verbose=True,
):
    """Create a multi-slice plot of image and return the saved file.

    Parameters
    ----------
    imagef : str
        The path to the image file to plot.
    subj : str
        The subject ID.
    tracer : str
        The PET tracer used.
    pet_date : str
        The PET date.
    display_mode : str, optional
        The display mode to use.
    cut_coords : list, optional
        The cut coordinates to use.
    annotate : bool, optional
        Whether to annotate the slices.
    draw_cross : bool, optional
        Whether to draw a cross on the slices.
    colorbar : bool, optional
        Whether to draw a colorbar.
    cbar_tick_format : str, optional
        The format to use for the colorbar ticks.
    figsize : tuple, optional
        The figure size to use.
    dpi : int, optional
        The DPI to use.
    font : dict, optional
        The font sizes to use.
    title : str, optional
        The title to use.
    fig : matplotlib.figure.Figure, optional
        The figure to use.
    ax : matplotlib.axes.Axes, optional
        The axes to use.
    overwrite : bool, optional
        Whether to overwrite the output file if it already exists.
    verbose : bool, optional
        Whether to print status messages.
    """
    tracer_fancy = {"FBB": "Florbetaben", "FBP": "Florbetapir"}[tracer]
    auto_ticks = (vmin is None) and (vmax is None)
    if tracer == "FBB":
        if cmap is None:
            cmap = "binary_r"
        if vmin is None or vmin == "auto":
            vmin = 0
        if vmax is None:
            vmax = 2.5
        elif vmax == "auto":
            vmax = np.nanmax(nops.load_nii_flex(imagef, dat_only=True))
        if auto_ticks:
            cbar_ticks = [0, 0.5, 1, 1.5, 2, 2.5]
        else:
            cbar_ticks = np.round(np.linspace(vmin, vmax, 5), 1)
        facecolor = "k"
        fontcolor = "w"
        black_bg = True
    elif tracer == "FBP":
        if cmap is None:
            cmap = "binary"
        if vmin is None or vmin == "auto":
            vmin = 0
        if vmax is None:
            vmax = 2.2
        elif vmax == "auto":
            vmax = np.nanmax(nops.load_nii_flex(imagef, dat_only=True))
        if auto_ticks:
            cbar_ticks = [0, 0.55, 1.1, 1.65, 2.2]
        else:
            cbar_ticks = np.round(np.linspace(vmin, vmax, 5), 1)
        facecolor = "w"
        fontcolor = "k"
        black_bg = False

    # Make the plot.
    plt.close("all")
    if fig is None or ax is None:
        fig, ax = plt.subplots(
            3, 1, height_ratios=[1.2, 6, 1.8], figsize=figsize, dpi=dpi
        )
        ax = np.ravel(ax)
    else:
        assert len(ax) == 3

    iax = 1
    _ax = ax[iax]
    warnings.filterwarnings("ignore", category=UserWarning)
    display = plotting.plot_anat(
        imagef,
        cut_coords=cut_coords,
        display_mode=display_mode,
        annotate=annotate,
        draw_cross=draw_cross,
        black_bg=black_bg,
        cmap=cmap,
        colorbar=colorbar,
        cbar_tick_format=cbar_tick_format,
        vmin=vmin,
        vmax=vmax,
        title=title,
        figure=fig,
        axes=_ax,
    )
    warnings.resetwarnings()

    # Add the colorbar.
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=_ax,
        location="bottom",
        pad=0.1,
        shrink=0.8,
        aspect=30,
        drawedges=False,
    )
    cbar.outline.set_color(fontcolor)
    cbar.outline.set_linewidth(1)
    cbar.ax.tick_params(labelsize=font["tick"], labelcolor=fontcolor)
    cbar.ax.set_xticks(cbar_ticks)
    cbar.ax.set_xlabel(
        f"{tracer_fancy} SUVR", fontsize=font["label"], color=fontcolor, labelpad=10
    )

    # Format the top and bottom of the figure.
    for iax in [0, 2]:
        _ax = ax[iax]
        _ax.axis("off")

    _ax = ax[1]
    _ax.set_title(
        f"{subj}: {tracer_fancy} {pet_date}", fontsize=font["title"], color=fontcolor
    )

    for iax in range(len(ax)):
        _ax = ax[iax]
        _ax.set_facecolor(facecolor)
    fig.patch.set_facecolor(facecolor)

    # Save the figure as a pdf.
    outfile = (
        strm.add_presuf(imagef, prefix="_", suffix="_multislice")
        .replace(".nii.gz", ".pdf")
        .replace(".nii", ".pdf")
    )
    if overwrite or not op.isfile(outfile):
        fig.savefig(
            outfile,
            facecolor=facecolor,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.2,
        )
        if verbose:
            print(f"  Saved {outfile}")
        return outfile
    else:
        return None


def merge_multislice(
    intermed_mslicef,
    template_dir,
    tracer,
    remove_intermed=True,
    overwrite=False,
    verbose=True,
):
    """Merge multislice PDF of a scan with the matching tracer template.

    Parameters
    ----------
    intermed_mslicef : str
        The multi-slice PDF to merge with the template.
    tracer : str
        The PET tracer.
    remove_intermed : bool
        Whether to remove the intermediate multislice file.
    overwrite : bool
        Whether to overwrite the output file if it already exists.
    verbose : bool
        Whether to print status messages.

    Returns
    -------
    outfile : str
        The merged PDF.
    """
    # Load the template PDF
    templatef = op.join(template_dir, f"ADNI4_{tracer}_template.pdf")

    # Get the output filename.
    outfile = op.join(op.dirname(intermed_mslicef), op.basename(intermed_mslicef)[1:])

    # Merge the PDFs.
    cmd = f"qpdf --linearize --qdf --optimize-images --empty --pages {templatef} 1 {intermed_mslicef} 1 -- {outfile}"
    _ = osu.run_cmd(cmd)
    if verbose:
        print(f"  Saved {outfile}")

    # writer = PyPDF2.PdfWriter()
    # for filename in [templatef, intermed_mslicef]:
    #     reader = PyPDF2.PdfReader(filename)
    #     page = reader.pages[0]
    #     page.scale_to(width=960, height=540)  # 13.33 x 7.5 inches
    #     page.compress_content_streams()
    #     writer.add_page(page)
    #     if filename == templatef:
    #         writer.add_metadata(reader.metadata)

    # Remove the intermediary file.
    if remove_intermed:
        os.remove(intermed_mslicef)
        if verbose:
            print(f"  Removed {intermed_mslicef}")

    # # Save the merged PDF.
    # if overwrite or not op.isfile(outfile):
    #     if op.isfile(outfile):
    #         os.remove(outfile)
    #     with open(outfile, "wb") as output:
    #         writer.write(output)
    #     if verbose:
    #         print(f"  Saved {outfile}")
    # else:
    #     return None

    # # Close file descriptors.
    # writer.close()

    # # Fix the file so Acrobat can open it.
    # _tmp_outfile = strm.add_presuf(outfile, prefix="__")

    # with pikepdf.Pdf.open(outfile) as pdf:
    #     pdf.save(_tmp_outfile, qdf=True)
    # os.remove(outfile)
    # os.rename(_tmp_outfile, outfile)

    return outfile


def _parse_args():
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(
        description="""Process ADNI amyloid PET scans for visual read.

steps:
  [1] Copy PET scan from [base_dir]/raw/[subject]/[nested_dirs_from_LONI]/[pet_scan].nii
      to [base_dir]/data/proc/sub-[subject]/pet-[tracer]/ses-[pet_date]/intermed/
      mean_sub-[subject]_pet-[tracer]_ses-[pet_date].nii
  [2] Reset origin to center (changing header info of the copied image)
  [3] Smooth PET to a specified resolution (default 8mm isotropic). Note that if the
      input and final resolutions are the same, smoothing is skipped
  [4] Coregister PET (default 6-degree rigid body transform) to a standard space
  [5] Save axial multislice images of the coregistered scan as a PDF, and merge
      this file with the canonical positive and negative scan template
      for the matching tracer.""",
        formatter_class=TextFormatter,
        exit_on_error=False,
    )
    parser.add_argument(
        "-d",
        "--base_dir",
        type=str,
        default="/Volumes/petcore/Projects/ADNI_Reads",
        help=(
            "Path to base directory where data/ and templates/ are stored.\n"
            + "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "-s",
        "--subjects",
        nargs="*",
        type=str,
        help=(
            "List of subjects to process. If --subjects is not specified and\n"
            + "--overwrite is not specified, then all unprocessed subjects in\n"
            + "[base_dir]/raw/ will be processed. If --overwrite is specified,\n"
            + "then all subjects in [base_dir]/raw/ will be processed."
        ),
    )
    parser.add_argument(
        "--skip_proc",
        action="store_true",
        help=(
            "Skip image processing and go straight to multislice creation\n"
            + "(only works if images have already been processed; otherwise\n"
            + "process_pet() is still called)"
        ),
    )
    parser.add_argument(
        "--use_spm",
        action="store_true",
        help="Use SPM for processing, instead of default FSL and niimath",
    )
    parser.add_argument(
        "--final_res",
        default=8,
        type=float,
        help="Final resolution (FWHM) of smoothed PET scans, in mm. Default: %(default)s",
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
        "--skip_multislice",
        action="store_true",
        help="Process PET but skip multislice PDF creation",
    )
    parser.add_argument(
        "-z",
        "--zslice",
        type=int,
        nargs="+",
        default=[
            -46,
            -32,
            -18,
            -4,
            10,
            24,
            38,
            52,
        ],  # [-51, -30, -21, -6, 9, 24, 39, 54],
        help=(
            "List of image slices to show along the z-axis, in MNI coordinates\n"
            + "Default: %(default)s"
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
        "--vmin",
        help=(
            "Minimum intensity threshold for the multislice images\n"
            + "(overrides the tracer-specific defaults). Can be a float or\n"
            + "'auto', in which case vmax is set to the maximum intensity"
        ),
    )
    parser.add_argument(
        "--vmax",
        help=(
            "Maximum intensity threshold for the multislice images\n"
            + "(overrides the tracer-specific defaults). Can be a float or\n"
            + "'auto', in which case vmax is set to the maximum intensity"
        ),
    )
    parser.add_argument(
        "-o", "--overwrite", action="store_true", help="Overwrite existing files"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Run without printing output"
    )

    # Parse the command line arguments
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    args = parser.parse_args()
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
    timer = Timer()

    # Get command line arguments.
    args = _parse_args()

    # Format arguments.
    proc_res = [args.final_res] * 3

    use_spm = False
    if args.use_spm:
        use_spm = True
        if args.coreg_dof != 6:
            print(
                "SPM only supports 6-degree coregistration; over-riding --coreg_dof to 6"
            )
            args.coreg_dof = 6

    verbose = True
    if args.quiet:
        verbose = False

    # Get the dataframe of scans to process.
    proc_df = find_scans_to_process(
        args.base_dir,
        process_subjs=args.subjects,
        proc_res=proc_res,
        overwrite=args.overwrite,
        save_output=True,
        verbose=verbose,
    )
    if verbose:
        print(
            "Selected {}/{} subjects to process, with {}/{} subjects already processed".format(
                proc_df["process"].sum(),
                len(proc_df),
                proc_df["processing_complete"].sum(),
                len(proc_df),
            )
        )

    # Process each scan and save a multislice PDF of the processed
    # image.
    for subj in proc_df.query("process == True").index:
        rsmean_petf = nops.find_gzip(
            proc_df.at[subj, "rsmean_petf"], return_infile=True
        )
        if op.isfile(rsmean_petf) and args.skip_proc:
            if verbose:
                print(
                    "  Skipping image processing and jumping to multislice PDF creation"
                )
        else:
            # Process the PET image.
            rsmean_petf = process_pet(
                base_dir=args.base_dir,
                raw_petf=proc_df.at[subj, "raw_petf"],
                mean_petf=proc_df.at[subj, "mean_petf"],
                tracer=proc_df.at[subj, "tracer"],
                input_res=proc_df.at[subj, "input_res"],
                proc_res=proc_df.at[subj, "proc_res"],
                coreg_dof=args.coreg_dof,
                use_spm=use_spm,
                verbose=verbose,
            )

        if args.skip_multislice:
            if verbose:
                print("  Skipping multislice PDF creation")
        else:
            # Create the multislice PDF.
            intermed_mslicef = create_multislice(
                imagef=rsmean_petf,
                subj=subj,
                tracer=proc_df.at[subj, "tracer"],
                pet_date=proc_df.at[subj, "pet_date"],
                cut_coords=args.zslice,
                cmap=args.cmap,
                vmin=args.vmin,
                vmax=args.vmax,
                overwrite=args.overwrite,
                verbose=False,
            )

            # Merge the subject's multislice images with the tracer
            # template.
            mslicef = merge_multislice(
                intermed_mslicef=intermed_mslicef,
                template_dir=op.join(args.base_dir, "templates"),
                tracer=proc_df.at[subj, "tracer"],
                remove_intermed=False,
                overwrite=args.overwrite,
                verbose=verbose,
            )

        # Create a visual read file for this subject.
        pass

        # Print the runtime for this subject.
        if verbose:
            print(timer.loop(f"  {subj}"))

    # Print the total runtime.
    if verbose:
        print(timer)
    sys.exit(0)
