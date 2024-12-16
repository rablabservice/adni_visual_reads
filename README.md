# Process ADNI Amyloid PET Scans for Visual Read

This code processes ADNI amyloid PET scans by converting DICOM files to NIFTI format, resetting the origin, and generating image slice visualizations in PDF format that can be compared against 'canonical' positive and negative scan templates. The code also supports various options for smoothing, coregistration, and cropping, and can handle multiple subjects in a batch process.

## Steps
1. Copy PET scan from `[base_dir]/data/[raw_dirname]/[subject]/[nested_dirs_from_LONI]/[pet_scan].nii` to `[base_dir]/data/[proc_dirname]/[subject]/[tracer]_[pet_date]/[subject]_[tracer]_[pet_date].nii`
2. Convert DICOMS to NIFTI (if scans are not already provided as NIFTIs)
3. Reset origin to center (saves over header info of the copied image)
4. Save a PDF of axial multislices of the processed PET scan and a merged PDF that also shows canonical positive and negative scans for the same tracer

## Output files
### For each processed scan
- Processed PET scan: `[subject]_[tracer]_[date].nii[.gz]`
- Multi-slice visualization: `[subject]_[tracer]_[date]_multislice.pdf`
- Merged template visualization: `[subject]_[tracer]_[date]_multislice_merged.pdf`

### Per code run
- Processing log: pet_proc_[timestamp].csv

## Options
- `-h, --help`: Show this help message and exit
- `--base_dir BASE_DIR`: Path to base directory where data/ and templates/ are stored (default: `/shared/petcore/Projects/ADNI_Reads`)
- `--raw_dirname RAW_DIRNAME`: Name of the subdirectory in `[base_dir]/data` where raw PET files (raw meaning we haven't done anything to them yet) for scans to process are stored (default: `raw`)
- `--proc_dirname PROC_DIRNAME`: Name of the subdirectory in `[base_dir]/data` where processed PET and multislice files are stored (default: `proc`)
- `-s [SUBJECTS ...], --subjects [SUBJECTS ...]`: List of subjects to process. If `--subjects` is not defined and `--overwrite` is not defined, then all unprocessed subjects in `[base_dir]/data/[raw_dirname]/` will be processed. If `--overwrite` is defined, then all subjects in `[base_dir]/data/raw/` will be processed
- `--smooth, --no-smooth`: Smooth PET to the resolution defined by `--final_res`
- `--final_res FINAL_RES`: Final resolution (FWHM) of smoothed PET, in mm (default: `6`)
- `--coreg, --no-coreg`: Coregister and reslice PET to standard space
- `--coreg_dof {3,6,9,12}`: Degrees of freedom used for linear coregistration
  - `3`: Translation only
  - `6`: Translation + rotation (rigid body)
  - `9`: Translation + rotation + scaling
  - `12`: Translation + rotation + scaling + shearing (full affine)
  - Default: `6`
- `--use_spm`: Use SPM for processing, instead of default FSL and niimath
- `--gzip, --no-gzip`: Gzip processed NIfTI files (raw files are untouched)
- `--skip_proc`: Skip PET processing and jump straight to multislice PDF creation. Requires PET to have already been processed, otherwise PET processing is still completed. Use this flag with `--overwrite` to keep PET processing untouched but overwrite multislice PDFs
- `--skip_multislice`: Process PET but skip multislice PDF creation
- `-z SLICES [SLICES ...], --slices SLICES [SLICES ...]`: List of image slices to show along the z-axis, in MNI coordinates (default: `[-50, -38, -26, -14, -2, 10, 22, 34]`)
- `--crop, --no-crop`: Crop the multislice images to the brain
- `--mask_thresh MASK_THRESH`: Cropping threshold for defining empty voxels outside the brain; used together with crop_prop to determine how aggressively to remove planes of mostly empty space around the image (default: `0.05`)
- `--crop_prop CROP_PROP`: Defines how tightly to crop the brain for multislice creation (proportion of empty voxels in each plane that are allowed to be cropped) (default: `0.05`)
- `--cmap CMAP`: Colormap to use for the multislice images (overrides the tracer-specific defaults)
- `--vmin VMIN`: Minimum intensity threshold for the multislice images (overrides the tracer-specific defaults)
- `--vmax VMAX`: Maximum intensity threshold for the multislice images (overrides the tracer-specific defaults)
- `--autoscale`: Autoscale vmax to a percentile of image values > 0 (see `--autoscale_max_pct`). Overridden by `--vmax`, so don't both setting both of these options
- `--autoscale_max_pct AUTOSCALE_MAX_PCT`: Set the percentile of included voxel values to use for autoscaling the maximum colormap intensity (vmax)
- `--n_cbar_ticks N_CBAR_TICKS`: Number of ticks to show for the multislice PDF colorbar (default: `2`)
- `-o, --overwrite`: Overwrite existing files
- `-q, --quiet`: Run without printing output
- `-d, --dry_run`: Show what scans would be processed but don't actually do anything