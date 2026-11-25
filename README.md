# TYROBP in ulcerative colitis: analysis code

Consolidated R and Python scripts for bulk transcriptomic, single-cell, and spatial transcriptomic analyses of TYROBP in ulcerative colitis.

## Files

| Script | Contents |
| --- | --- |
| `transcriptome_R.R` | Bulk transcriptomic plots and analyses, including PCA, differential-expression visualization, WGCNA result plots, enrichment, feature selection, and immune-cell analyses. |
| `transcriptome_python.py` | Four-gene model plots, UC-versus-CD analyses, and clinical activity/treatment-response plots. |
| `singlecell_spatial.R` | Single-cell annotation and QC plots, CellChat and pseudotime plots, donor-level pseudobulk analyses, and macrophage-score-adjusted TYROBP expression. |
| `singlecell_spatial.py` | Spatial expression/signature maps, donor composition and expression plots, and macrophage-subset plots. |

The scripts retain recovered native plotting functions and expose individual stages through command-line entry points. Historical source code is embedded where needed.

## List available stages

```bash
Rscript transcriptome_R.R --list
Rscript singlecell_spatial.R --list
python transcriptome_python.py --list
python singlecell_spatial.py --list
```

## Inputs and reproducibility

**This repository contains code, not the complete input-data bundles. Cloning the repository alone is insufficient to reproduce all analyses.**

Depending on the stage, required inputs include saved expression/result tables, sample annotations, native plot caches, fitted CellChat/Slingshot objects, and spatial tissue images with spot coordinates. These must be supplied in the corresponding data layout before running a stage. The original data bundles and their documentation describe those layouts and provenance.

For the single-cell/spatial Python integrity check, `provenance/input_sha256.json` is also required beside the script. Providing `--data-root` alone does not supply that manifest.

Example commands after placing the corresponding data bundle beside the scripts:

```bash
Rscript transcriptome_R.R --check
Rscript transcriptome_R.R --stage fig3_lasso
Rscript singlecell_spatial.R --stage annotation_myeloid
python transcriptome_python.py --check
python singlecell_spatial.py --stage fig03_spatial --preview-only
```

Use `--data-root` and `--output-root` to select input and output locations. Refer to each script's stage list and help text for stage-specific requirements. Some archived source blocks retain historical local paths; the consolidated entry points provide the intended execution interface.

The scripts distinguish replotting saved results from rerunning upstream analyses. Some original interactive-app settings and the pre-QC Seurat checkpoint were not recovered. External-object entry points remain available. The macrophage-subset script provides both the earlier orange-red palette option and the saved purple version; the exact warm-version source was not recovered. Sparse macrophage subsets remain descriptive, with missing values retained.

## Software

The prepared scripts were checked in R 4.5.1 and Python 3.13.12. Python dependencies are listed in `requirements_python.txt`. R dependencies vary by stage and include Seurat, edgeR, limma, ggplot2, WGCNA, clusterProfiler, glmnet, caret, randomForest, CellChat, ComplexHeatmap, slingshot, plot1cell, and plotting/export packages. The R `--check` commands report stage requirements.

Arial and Times New Roman were used in the original figures; font availability can affect rendering.

## Source notices

Original third-party author and copyright notices embedded in recovered source blocks, including CIBERSORT notices, are retained. No additional blanket license is assigned by this repository.
