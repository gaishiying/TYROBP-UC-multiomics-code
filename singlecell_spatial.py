"""Reproduce the supplied 3 x 4 spatial transcriptomics figure from genuine spot tables.

The panel function, palette, scales, fonts, layout and conditional coloring are
adapted from redraw_figure8_no_black_boxes.py (2026-11-24). The original source
and its preprocessing scripts are preserved separately for provenance. This
offline stage reads saved, quality-controlled data; it does not claim to rerun
raw count preprocessing or reference mapping.

Examples:
  python spatial_sections.py --list
  python spatial_sections.py --check --data-dir data/spatial
  python spatial_sections.py --stage fig03_spatial --data-dir data/spatial --preview-only
  python spatial_sections.py --stage fig03_spatial --data-dir data/spatial --dpi 900
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

SPATIAL_SECTIONS = ("B10", "C5", "B8", "B12")
SPATIAL_SIGNATURE = "Myeloid macrophage signature"
SPATIAL_SIGNATURE_MARKERS = (
    "LYZ", "CD68", "AIF1", "LST1", "FCGR3A", "FCGR1A", "CSF1R",
    "ITGAM", "MS4A7", "C1QA", "C1QB", "C1QC", "CD14",
)
SPATIAL_DEPENDENCIES = {"numpy": "numpy", "pandas": "pandas", "PIL": "Pillow", "matplotlib": "matplotlib"}


def _spatial_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _spatial_required_input_paths(data_root):
    base = Path(data_root)
    paths = [base / "tables" / "Plot_parameters.json", base / "tables" / "sample_metadata.csv"]
    for section in SPATIAL_SECTIONS:
        paths.extend((
            base / "tables" / f"{section}_plot_data.csv.gz",
            base / "qc_tables" / f"{section}_spot_metrics.csv.gz",
            base / "spatial" / section / "scalefactors_json.json",
            base / "spatial" / section / "tissue_hires_image.png",
            base / "spatial" / section / "tissue_positions_list.csv",
        ))
    return paths


def _spatial_load_validated_inputs(data_root):
    import numpy as np
    import pandas as pd
    from PIL import Image

    base = Path(data_root).resolve()
    missing = [str(p) for p in _spatial_required_input_paths(base) if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing spatial input(s):\n" + "\n".join(missing))
    params = json.loads((base / "tables" / "Plot_parameters.json").read_text(encoding="utf-8"))
    meta = pd.read_csv(base / "tables" / "sample_metadata.csv").set_index("sample_id")
    assert len(meta.loc[list(SPATIAL_SECTIONS), "patient_id"].unique()) == 4, "Expected four distinct donors"
    assert meta.loc[list(SPATIAL_SECTIONS), "group"].tolist() == ["Control", "Control", "UC", "UC"]
    assert list(params["main_sections"]) == list(SPATIAL_SECTIONS)
    assert "TYROBP" not in SPATIAL_SIGNATURE_MARKERS
    data, assets, verification = {}, {}, []
    for section in SPATIAL_SECTIONS:
        p = base / "tables" / f"{section}_plot_data.csv.gz"
        d = pd.read_csv(p)
        old_all = pd.read_csv(base / "qc_tables" / f"{section}_spot_metrics.csv.gz")
        old = old_all.query("pass_QC == True").set_index("barcode")
        assert d.barcode.is_unique and old.index.is_unique
        assert len(old) == len(d) and set(old.index) == set(d.barcode)
        for column in ("TYROBP_raw_count", "TYROBP_log1p_CP10k", SPATIAL_SIGNATURE, "pixel_col", "pixel_row"):
            assert np.allclose(d[column], old.loc[d.barcode, column], rtol=1e-12, atol=1e-12), (section, column)
        # Confirm recorded QC flags against the original thresholds, without changing inclusion.
        expected_qc = ((old_all.in_tissue == 1) & (old_all.detected_genes > 200)
                       & (old_all.total_UMI > 500) & (old_all.mitochondrial_pct < 30))
        assert np.array_equal(old_all.pass_QC.to_numpy(), expected_qc.to_numpy())
        expected_gene = np.log1p(10000.0 * d.TYROBP_raw_count.to_numpy() / d.total_UMI.to_numpy())
        assert np.allclose(d.TYROBP_log1p_CP10k, expected_gene, rtol=1e-12, atol=1e-12)
        image_path = base / "spatial" / section
        sf = json.loads((image_path / "scalefactors_json.json").read_text(encoding="utf-8"))
        with Image.open(image_path / "tissue_hires_image.png") as tissue:
            im = np.asarray(tissue).copy()
        pos = pd.read_csv(image_path / "tissue_positions_list.csv", header=None,
                          names=["barcode", "in_tissue", "array_row", "array_col", "pixel_row", "pixel_col"])
        assert pos.barcode.is_unique
        positions = pos.set_index("barcode").loc[d.barcode]
        for coordinate in ("pixel_col", "pixel_row", "array_col", "array_row", "in_tissue"):
            assert np.array_equal(positions[coordinate].to_numpy(), d[coordinate].to_numpy()), (section, coordinate)
        xy = d[["pixel_col", "pixel_row"]].to_numpy() * sf["tissue_hires_scalef"]
        assert np.isfinite(xy).all() and np.all(xy >= 0)
        assert xy[:, 0].max() <= im.shape[1] and xy[:, 1].max() <= im.shape[0], "Coordinates exceed hires tissue image"
        data[section], assets[section] = d, (im, sf)
        verification.append({
            "section": section, "group": str(meta.loc[section, "group"]),
            "patient_id": str(meta.loc[section, "patient_id"]),
            "geo_accession": str(meta.loc[section, "geo_accession"]),
            "spots": int(len(d)), "TYROBP_positive_spots": int((d.TYROBP_raw_count > 0).sum()),
            "tissue_image_width_height": [int(im.shape[1]), int(im.shape[0])],
            "hires_scale": float(sf["tissue_hires_scalef"]),
            "spot_diameter_fullres": float(sf["spot_diameter_fullres"]),
            "scaled_pixel_col_range": [float(xy[:, 0].min()), float(xy[:, 0].max())],
            "scaled_pixel_row_range": [float(xy[:, 1].min()), float(xy[:, 1].max())],
            "source_plot_table_sha256": _spatial_sha256(p),
            "QC_table_and_original_tissue_position_match": True,
            "TYROBP_normalization_formula_match": True,
        })
    return base, params, meta, data, assets, verification


def check_spatial_inputs(data_root=Path("data/spatial"), font_dir=None):
    """Validate dependencies, genuine data, QC, barcodes, coordinates, scales and donors."""
    unavailable = [package for module, package in SPATIAL_DEPENDENCIES.items() if importlib.util.find_spec(module) is None]
    if unavailable:
        raise ImportError("Install spatial plotting dependencies: " + ", ".join(unavailable))
    base, params, meta, data, assets, verification = _spatial_load_validated_inputs(data_root)
    return {
        "stage": "fig03_spatial", "status": "passed", "data_root": str(base),
        "dependencies": SPATIAL_DEPENDENCIES, "sections": verification,
        "distinct_donors": int(meta.loc[list(SPATIAL_SECTIONS), "patient_id"].nunique()),
        "gene_scale": params["gene_scale"], "signature_scale": params["marker_scale"],
        "input_sha256": {str(p.relative_to(base)): _spatial_sha256(p) for p in _spatial_required_input_paths(base)},
    }


def run_spatial_stage(data_root=Path("data/spatial"), output_root=Path("figures/spatial"),
                      dpi=900, preview_only=False, font_dir=None):
    """Run fig03_spatial; retain the original native Matplotlib panel function/style.

    `preview_only` exports the original 112 dpi PNG preview and validation report.
    Otherwise also exports a JPEG at requested dpi (original default 900 dpi).
    The display caps were pooled across all nine original sections and are read
    as saved; they are not recalculated from this four-section subset.
    """
    import numpy as np
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle
    from matplotlib.colors import Normalize, LinearSegmentedColormap

    if int(dpi) <= 0:
        raise ValueError("dpi must be a positive integer")
    base, params, meta, data, assets, verification = _spatial_load_validated_inputs(data_root)
    out = Path(output_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    font_dir = Path(font_dir) if font_dir else Path("C:/Windows/Fonts")
    for name in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
        font_path = font_dir / name
        if font_path.is_file():
            font_manager.fontManager.addfont(str(font_path))
    fontfile = font_manager.findfont(font_manager.FontProperties(family="Times New Roman", weight="normal"),
                                    fallback_to_default=False)
    # Preserve the original font; explicitly provide --font-dir on another OS.
    assert Path(fontfile).name.lower() == "times.ttf", "Use the original Times New Roman font files"
    style = {"font.family": "Times New Roman", "font.size": 12, "pdf.fonttype": 42, "ps.fonttype": 42,
             "savefig.facecolor": "white", "figure.facecolor": "white"}
    cmap = LinearSegmentedColormap.from_list("Original_Seurat_SpatialColors", params["palette"], N=256)
    norms = {"gene": Normalize(*params["gene_scale"], clip=True),
             "marker": Normalize(*params["marker_scale"], clip=True),
             "conditional": Normalize(*params["marker_scale"], clip=True)}

    def panel(ax, section, kind):
        # Original native plotting function, unchanged apart from variable names.
        d = data[section]
        im, sf = assets[section]
        scale = sf["tissue_hires_scalef"]
        xy = d[["pixel_col", "pixel_row"]].to_numpy() * scale
        radius = sf["spot_diameter_fullres"] * scale * .47
        circles = [Circle(tuple(p), radius) for p in xy]
        ax.imshow(im, alpha=.85)
        if kind == "conditional":
            ax.add_collection(PatchCollection(circles, facecolor="#C7C7C7", edgecolor="white",
                                              linewidth=.08, alpha=.48, rasterized=True))
            positive = d.TYROBP_raw_count.to_numpy() > 0
            circles = [p for p, keep in zip(circles, positive) if keep]
            values = d.loc[positive, SPATIAL_SIGNATURE].to_numpy()
        else:
            values = d["TYROBP_log1p_CP10k" if kind == "gene" else SPATIAL_SIGNATURE].to_numpy()
        collection = PatchCollection(circles, cmap=cmap, norm=norms[kind], edgecolors="white",
                                     linewidth=.08, alpha=1, rasterized=True)
        collection.set_array(values)
        ax.add_collection(collection)
        pad = 2.6 * radius
        ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
        ax.set_ylim(xy[:, 1].max() + pad, xy[:, 1].min() - pad)
        # As in the supplied final figure, no ROI rectangle annotations are added.
        ax.set_aspect("equal")
        ax.axis("off")
        return collection

    with plt.rc_context(style):
        fig = plt.figure(figsize=(14.4, 11.8))
        grid = fig.add_gridspec(3, 4, left=.055, right=.985, bottom=.012, top=.895, hspace=.30, wspace=.065)
        axes = np.array([[fig.add_subplot(grid[r, c]) for c in range(4)] for r in range(3)])
        rowtitles = ("TYROBP", "Macrophage-related signature", "Macrophage-related signature at TYROBP+ spots")
        for row, kind in enumerate(("gene", "marker", "conditional")):
            for column, section in enumerate(SPATIAL_SECTIONS):
                pc = panel(axes[row, column], section, kind)
                if row == 0:
                    axes[row, column].set_title(f'{meta.loc[section, "group"]}  {section}',
                                               fontsize=16, fontweight="bold", pad=9)
            top = grid[row, 0].get_position(fig).y1
            fig.text(.022, top + .040, "ABC"[row], fontfamily="Times New Roman", fontweight="normal",
                     fontsize=26, ha="left", va="center")
            fig.text(.060, top + .040, rowtitles[row], fontsize=17, fontweight="bold", ha="left", va="center")
            ca = fig.add_axes([.76, top + (.051 if row == 0 else .022), .205, .012])
            cb = fig.colorbar(pc, cax=ca, orientation="horizontal", extend="max")
            cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize=12, length=2, pad=2)
            cb.ax.set_title("ln(1 + CP10k)" if row == 0 else "Mean marker ln(1 + CP10k)", fontsize=13, pad=7)
            cb.set_ticks([0, .5, 1, 1.5, 2, 2.5] if row == 0 else [0, .2, .4, .6, .8])
        assert len(fig.axes) == 15 and fig._suptitle is None and len(fig.texts) == 6
        assert all(len(ax.patches) == 0 for ax in axes.flat)
        outputs = []
        preview_path = out / "Figure8_no_black_boxes_preview.png"
        fig.savefig(preview_path, bbox_inches="tight", pad_inches=.08, dpi=112)
        outputs.append(str(preview_path))
        if not preview_only:
            stem = f"Figure8_no_black_boxes_{int(dpi)}dpi"
            jpg_path = out / f"{stem}.jpg"
            fig.savefig(jpg_path, bbox_inches="tight", pad_inches=.08, dpi=int(dpi),
                        pil_kwargs={"quality": 98, "subsampling": 0, "dpi": (int(dpi), int(dpi))})
            with Image.open(jpg_path) as im:
                assert im.format == "JPEG" and im.mode == "RGB"
                assert im.info.get("dpi") and min(im.info["dpi"]) == int(dpi)
            outputs.append(str(jpg_path))
        plt.close(fig)
    report = {
        "stage": "fig03_spatial", "status": "passed", "data_root": str(base), "outputs": outputs,
        "reproduction_scope": "offline replot from original saved spot tables and tissue images",
        "raw_preprocessing_rerun": False, "original_plot_script": "redraw_figure8_no_black_boxes.py",
        "plotting_library": "Matplotlib", "matplotlib_version": matplotlib.__version__, "font_file": fontfile,
        "sections": verification, "four_distinct_donors": True,
        "display_scales": {"gene": params["gene_scale"], "marker": params["marker_scale"]},
        "display_caps_origin": "saved pooled 99th percentiles across nine original retained-spot sections",
        "target_positive_definition": "Raw TYROBP UMI > 0",
        "normalization": "natural log(1 + 10000 * raw gene count / spot total UMI)",
        "macrophage_signature": "arithmetic mean of 13 marker ln(1+CP10k) values; TYROBP excluded",
        "macrophage_signature_markers": list(SPATIAL_SIGNATURE_MARKERS),
        "no_target_smoothing_or_imputation": True, "spot_level_inferential_P_values": False,
        "ROI_rectangle_count": 0, "original_layout_inches": [14.4, 11.8],
        "preview_only": bool(preview_only), "JPEG_requested_dpi": None if preview_only else int(dpi),
    }
    (out / "Figure8_no_black_boxes_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


PY_STAGES = {"fig03_spatial": run_spatial_stage}


def run_python_stage(stage, data_root=Path("data/spatial"), output_root=Path("figures/spatial"), **kwargs):
    if stage not in PY_STAGES:
        raise ValueError(f"Unknown stage {stage!r}; available: {', '.join(PY_STAGES)}")
    return PY_STAGES[stage](data_root=data_root, output_root=output_root, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List registered spatial stages")
    parser.add_argument("--check", action="store_true", help="Validate data without rendering")
    parser.add_argument("--stage", choices=tuple(PY_STAGES), default="fig03_spatial")
    parser.add_argument("--data-dir", "--data-root", dest="data_root", type=Path, default=Path("data/spatial"))
    parser.add_argument("--output-dir", "--output-root", dest="output_root", type=Path, default=Path("figures/spatial"))
    parser.add_argument("--dpi", type=int, default=900, help="Original high-resolution JPEG export defaults to 900 dpi")
    parser.add_argument("--preview-only", action="store_true", help="Render only the original 112 dpi preview")
    parser.add_argument("--font-dir", type=Path, default=None, help="Folder containing the original Times New Roman TTF files")
    args = parser.parse_args()
    if args.list:
        print(json.dumps({"PY_STAGES": list(PY_STAGES)}, indent=2))
    elif args.check:
        print(json.dumps(check_spatial_inputs(args.data_root, args.font_dir), indent=2))
    else:
        print(json.dumps(run_python_stage(args.stage, args.data_root, args.output_root,
                                         dpi=args.dpi, preview_only=args.preview_only, font_dir=args.font_dir), indent=2))




# Consolidated native plotting code: donor and macrophage panels.
# Original function bodies are embedded below; only paths/entry points are unified.
from pathlib import Path
import sys, types, contextlib

_ROOT_SOURCES = {
'total_violin': '"""Redraw frozen patient-level TYROBP values in the supplied violin style.\n\nRun: python code/redraw_publication_violins.py [package_directory]\nOnly display styling changes. Each point is one observed donor–population pair.\nNo values are imputed, and no statistical test is recalculated.\n"""\nfrom pathlib import Path\nimport sys,json,hashlib,html,platform\nimport numpy as np\nimport pandas as pd\nfrom scipy.stats import gaussian_kde\nimport scipy\nimport matplotlib\nmatplotlib.use(\'Agg\')\nimport matplotlib.pyplot as plt\nfrom matplotlib.ticker import MultipleLocator,FormatStrFormatter\n\nROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1]\nGROUPS=[\'Control\',\'UC\']\nNAVY,WINE,INK=\'#173F63\',\'#AF1D40\',\'#202B34\'\nSUBSETS=[\'C1Q_mature\',\'ACOD1_inflammatory_like\',\'NRG1_activated_like\']\nTOTAL=\'Combined_macrophage_like\'\nTITLES={TOTAL:\'Total macrophages\',\'C1Q_mature\':\'C1Q+ mature macrophages\',\n \'ACOD1_inflammatory_like\':\'ACOD1+ inflammatory\\nmacrophage-like cells\',\n \'NRG1_activated_like\':\'NRG1+ activated\\nmacrophage-like cells\'}\nMANIFEST=[];PLOT_AUDIT=[]\n\ndef theme():\n plt.rcParams.update({\'font.family\':\'Arial\',\'font.size\':10,\'axes.titlesize\':11.5,\n  \'axes.labelsize\':10.5,\'axes.edgecolor\':INK,\'axes.labelcolor\':INK,\'text.color\':INK,\n  \'axes.linewidth\':.95,\'xtick.color\':INK,\'ytick.color\':INK,\'xtick.labelsize\':10,\n  \'ytick.labelsize\':10,\'xtick.major.size\':0,\'ytick.major.size\':3.2,\n  \'ytick.major.width\':.8,\'pdf.fonttype\':42,\'svg.fonttype\':\'none\',\n  \'mathtext.fontset\':\'dejavusans\',\'figure.facecolor\':\'white\',\'savefig.facecolor\':\'white\'})\n\ndef violin(ax,df,scope,stat,letter=None):\n d=df[df.scope.eq(scope)]\n arrays=[d.loc[d.group.eq(g),\'TMM_log2CPM\'].to_numpy() for g in GROUPS]\n joint=np.concatenate(arrays);span=max(float(np.ptp(joint)),1.)\n low=joint.min()-.18*span;high=joint.max()+.34*span\n rng=np.random.default_rng(20261117+sum(map(ord,scope)))\n for x,(v,color,g) in enumerate(zip(arrays,[NAVY,WINE],GROUPS)):\n  if len(v)==0:\n   ax.text(x,low+.48*(high-low),\'No observed\\ncells\',ha=\'center\',va=\'center\',\n            fontsize=9.5,color=\'#5C6770\')\n   continue\n  # A narrow KDE gives the compact horizontal bodies in the supplied example.\n  # Density is truncated at observed min/max. The KDE does not enter any test.\n  if len(v)>=3 and np.ptp(v)>1e-12:\n   kde=gaussian_kde(v,bw_method=.28)\n   grid=np.linspace(v.min(),v.max(),401)\n   den=kde(grid);half=.365*den/den.max()\n   ax.fill_betweenx(grid,x-half,x+half,facecolor=color,edgecolor=color,lw=.7,zorder=1)\n   # Small white box and median, close to the reference figure.\n   q1,median,q3=np.quantile(v,[.25,.5,.75])\n   ax.plot([x,x],[v.min(),v.max()],color=INK,lw=.65,zorder=2)\n   ax.add_patch(plt.Rectangle((x-.055,q1),.11,q3-q1,facecolor=\'white\',\n               edgecolor=INK,lw=.75,zorder=3))\n   ax.plot([x-.055,x+.055],[median,median],color=INK,lw=1.25,zorder=4)\n  # Fixed offsets keep each observed point visible, including small groups.\n  offsets=np.linspace(-.105,.105,len(v)) if len(v)>1 else np.array([0.])\n  rng.shuffle(offsets)\n  ax.scatter(x+offsets,v,s=35,facecolor=color,edgecolor=\'white\',linewidth=.85,zorder=5)\n  for row,xx,yy in zip(d[d.group.eq(g)].itertuples(),x+offsets,v):\n   PLOT_AUDIT.append(dict(scope=scope,group=g,donor=row.donor,x=float(xx),TMM_log2CPM=float(yy)))\n ax.set_xlim(-.55,1.55);ax.set_ylim(low,high)\n ax.set_xticks([0,1],[f\'{g}\\n(n = {len(v)})\' for g,v in zip(GROUPS,arrays)])\n ax.set_ylabel(\'TYROBP expression (log$_2$ CPM)\',labelpad=6)\n ax.yaxis.set_major_locator(MultipleLocator(.5 if high-low<2.8 else 1.))\n ax.yaxis.set_major_formatter(FormatStrFormatter(\'%.1f\'))\n ax.spines[[\'top\',\'right\']].set_visible(False);ax.grid(False)\n title=TITLES[scope]\n ax.set_title(title,pad=12,fontweight=\'normal\')\n if pd.notna(stat.PValue):\n  y=joint.max()+.17*span;h=.035*span\n  ax.plot([0,0,1,1],[y-h,y,y,y-h],color=INK,lw=.85)\n  ax.text(.5,y+.035*span,rf\'$P = {stat.PValue:.3f}$\',ha=\'center\',va=\'bottom\',fontsize=10.5)\n else:\n  ax.text(.5,.965,\'Descriptive only\',transform=ax.transAxes,ha=\'center\',va=\'top\',\n          fontsize=9,color=\'#5C6770\')\n if letter:\n  ax.text(-.21,1.13,letter,transform=ax.transAxes,fontweight=\'bold\',fontsize=14,\n          va=\'bottom\',ha=\'left\',clip_on=False)\n\ndef save(fig,folder,stem,description):\n out=ROOT/\'figures\'/folder;out.mkdir(parents=True,exist_ok=True)\n for ext in [\'png\',\'pdf\',\'svg\']:\n  fig.savefig(out/f\'{stem}.{ext}\',dpi=600,bbox_inches=\'tight\',pad_inches=.08)\n fig.savefig(out/f\'{stem}_preview.png\',dpi=160,bbox_inches=\'tight\',pad_inches=.08)\n MANIFEST.append(dict(folder=folder,stem=stem,description=description))\n plt.close(fig)\n\ndef documentation(df,stats):\n guide=\'\'\'TYROBP 发表用小提琴图（2026-11-17）\n\n正文：figures/01_main/Main_Total_macrophages_TYROBP_violin\n补充图三联图：figures/02_supplementary/Supplementary_Macrophage_subsets_TYROBP_violin\n三个亚群的单图：figures/03_individual_subsets\n各图提供 600 dpi PNG、矢量 PDF/SVG，以及预览 PNG。\n\n图形修改\n参照用户提供的样式：深蓝/酒红小提琴、白边散点、窄白色四分位箱体、左/下坐标轴。\n横轴为 Control / UC 和 (n = 数量)，已移除 donors 字样；顶部为对应细胞群标题。\n图中每个点仍代表一位实际检出该亚群的供者。n 为实际供者数，不是细胞数。\n总巨噬细胞为 C1Q+、ACOD1+、NRG1+ 三类细胞合并，共 596 个细胞，Control 167 / UC 429；各 6 位供者。\nC1Q 为 6/6；ACOD1 为 0/6；NRG1 为 2/3。\nACOD1 对照没有检出细胞；NRG1 对照只有 4 个细胞（来自两人）。缺失亚群不补成零表达。\n少于 3 个点不估计小提琴密度或绘制箱体，NRG1 对照只显示两个真实点。\nACOD1、NRG1 仅描述，保留 Descriptive only；不标注组间 P 值。\n\n数据和统计\n从 TYROBP_Macrophages_2026-11-17 的已验证表逐值重画，表达量和检验结果不变。\n总巨噬细胞 edgeR P = 0.4748183814（图上 P = 0.475），BH FDR = 0.7549201161。\nC1Q edgeR P = 0.8464291028（图上 P = 0.846），BH FDR = 0.9675801106。\n参考图中的旧 P = 0.04 和旧表达值未用于本次数据图。\n纵轴为供者 pseudobulk 的 TMM log2 CPM，prior.count=2。可检验群经 filterByExpr、TMM 及 robust edgeR 准似然检验，模型 ~group。\nP 为原始 edgeR P，统计表另列每比较内全部保留基因的 BH FDR。\nACOD1、NRG1 采用实际有细胞供者的描述性 TMM（全部 33538 测量基因，不做组间检验）。各群分别归一化，不据纵轴跨亚群直接比较。\n小提琴 KDE 仅显示分布，bw_method=0.28，截断于实际观察范围，不参与统计检验。\n完整统计复算代码和原始聚合输入保留于桌面上一版 TYROBP_Macrophages_2026-11-17 包；本包为轻量作图版本。\n\n复现图形\npython code/redraw_publication_violins.py\n依赖见 requirements.txt；data/ 包含作图数据与统计摘要，validation/ 保存输入散列及绘图点核验。\n\'\'\'\n captions=\'\'\'Main figure — English\nTYROBP expression in total macrophages. Total macrophages comprise the combined C1Q+, ACOD1+, and NRG1+ populations. Each point represents one donor; n indicates the number of donors with observed cells. Expression is shown as TMM-normalized log2 CPM. P is from the edgeR quasi-likelihood test.\n\n正文图 — 中文\n总巨噬细胞中的 TYROBP 表达。总巨噬细胞为 C1Q+、ACOD1+ 和 NRG1+ 三类细胞合并。每个散点代表一位供者，n 表示检出相应细胞群的供者数。表达量以 TMM 归一化 log2 CPM 展示。P 值来自 edgeR 准似然检验。\n\nSupplementary figure — English\nTYROBP expression across macrophage subsets. (A) C1Q+ mature macrophages. (B) ACOD1+ inflammatory macrophage-like cells. (C) NRG1+ activated macrophage-like cells. Each point represents one donor; n indicates the number of donors with observed cells. P is from the edgeR quasi-likelihood test. Panels B and C are descriptive only owing to absent or sparse control populations. Missing donor–population combinations were not assigned zero expression.\n\n补充图 — 中文\n各巨噬细胞亚群中的 TYROBP 表达。（A）C1Q+ 成熟巨噬细胞；（B）ACOD1+ 炎症性巨噬细胞样细胞；（C）NRG1+ 活化巨噬细胞样细胞。每个散点代表一位供者，n 表示检出相应细胞群的供者数。P 值来自 edgeR 准似然检验。B、C 因缺少或仅有极少对照细胞，仅作描述性展示。缺失的供者–亚群组合未记作零表达。\n\'\'\'\n (ROOT/\'README_图形说明.txt\').write_text(guide,encoding=\'utf-8-sig\')\n (ROOT/\'Figure_legends_ZH_EN.txt\').write_text(captions,encoding=\'utf-8-sig\')\n gallery=[]\n for item in MANIFEST:\n  base=f\'figures/{item["folder"]}/{item["stem"]}\'\n  links=\' · \'.join(f\'<a href="{base}.{ext}">{ext.upper()}</a>\' for ext in [\'png\',\'pdf\',\'svg\'])\n  gallery.append(f\'<article><h2>{html.escape(item["description"])}</h2><img src="{base}_preview.png"><p>{links}</p></article>\')\n page=\'\'\'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TYROBP 正文与补充图</title><style>body{margin:0;background:#f4f6f8;color:#243341;font:16px/1.7 Arial,"Microsoft YaHei",sans-serif}main{max-width:1100px;margin:auto;padding:30px}h1{font-size:27px}h2{font-size:19px}article{background:white;padding:22px;margin:24px 0;border-radius:7px}img{max-width:100%;height:auto}.note{border-left:4px solid #173f63;background:white;padding:18px}a{color:#174e79}pre{white-space:pre-wrap;font:14px/1.7 Arial,"Microsoft YaHei",sans-serif;background:white;padding:20px}</style><main><h1>TYROBP：正文总巨噬细胞图与补充亚群图</h1><p>已按参考样式重画；横轴保留组名和 n，顶部增加细胞群标题。</p><div class="note">数据和检验结果沿用已验证的供者层面分析：总巨噬细胞 P = 0.475，C1Q P = 0.846。ACOD1、NRG1 仅作描述性展示。n 为实际贡献表达的供者数。</div><p><a href="Figure_legends_ZH_EN.txt">中英文图注</a> · <a href="data/TYROBP_scope_results.csv">统计摘要</a> · <a href="README_图形说明.txt">作图说明</a></p>\'\'\'+\'\'.join(gallery)+\'<h2>使用与复现</h2><pre>\'+html.escape(guide)+\'</pre></main></html>\'\n (ROOT/\'TYROBP_正文与补充图预览.html\').write_text(page,encoding=\'utf-8\')\n\ndef main():\n theme()\n files=[ROOT/\'data/TYROBP_donor_expression.csv\',ROOT/\'data/TYROBP_scope_results.csv\']\n before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}\n df=pd.read_csv(files[0]);stats=pd.read_csv(files[1])\n assert len(df)==35 and set(df.scope)==set(SUBSETS+[TOTAL])\n assert np.isfinite(df.TMM_log2CPM).all() and df.n_cells.gt(0).all()\n def rs(scope):return stats[stats.scope.eq(scope)].iloc[0]\n for scope in SUBSETS+[TOTAL]:\n  row=rs(scope)\n  assert df[df.scope.eq(scope)&df.group.eq(\'Control\')].shape[0]==row.Control_donors\n  assert df[df.scope.eq(scope)&df.group.eq(\'UC\')].shape[0]==row.UC_donors\n  if scope in SUBSETS[1:]:assert pd.isna(row.PValue)\n # Main figure: only the combined population.\n fig,ax=plt.subplots(figsize=(3.7,3.75))\n violin(ax,df,TOTAL,rs(TOTAL));fig.tight_layout(pad=.65)\n save(fig,\'01_main\',\'Main_Total_macrophages_TYROBP_violin\',\'正文：总巨噬细胞\')\n for i,scope in enumerate(SUBSETS,1):\n  fig,ax=plt.subplots(figsize=(3.7,3.95))\n  violin(ax,df,scope,rs(scope));fig.tight_layout(pad=.65)\n  save(fig,\'03_individual_subsets\',f\'{i:02d}_{scope}_TYROBP_violin\',TITLES[scope].replace(\'\\n\',\' \'))\n fig,axes=plt.subplots(1,3,figsize=(11.2,4.05))\n for i,(ax,scope) in enumerate(zip(axes,SUBSETS)):\n  violin(ax,df,scope,rs(scope),chr(65+i))\n fig.subplots_adjust(left=.075,right=.99,bottom=.18,top=.78,wspace=.43)\n save(fig,\'02_supplementary\',\'Supplementary_Macrophage_subsets_TYROBP_violin\',\'补充图：三个亚群\')\n pd.DataFrame(MANIFEST).to_csv(ROOT/\'data/figure_manifest.csv\',index=False,encoding=\'utf-8-sig\')\n audit=pd.DataFrame(PLOT_AUDIT)\n # Each scope appears twice (standalone plus assembled supplement), except total.\n expected={TOTAL:1,**{scope:2 for scope in SUBSETS}}\n for scope,mult in expected.items():\n  actual=audit[audit.scope.eq(scope)]\n  assert len(actual)==len(df[df.scope.eq(scope)])*mult\n  for donor,rows in actual.groupby(\'donor\'):\n   original=df[df.scope.eq(scope)&df.donor.eq(donor)].TMM_log2CPM.item()\n   assert len(rows)==mult and np.allclose(rows.TMM_log2CPM,original,rtol=0,atol=0)\n for p in files:assert before[p.name]==hashlib.sha256(p.read_bytes()).hexdigest()\n audit.to_csv(ROOT/\'validation/plotted_points.csv\',index=False)\n validation={\'source_tables_unchanged\':True,\'all_plotted_values_match_inputs\':True,\n  \'horizontal_axis\':\'Control / UC and (n = ...); no donors label\',\n  \'titles\':TITLES,\'Total_P\':float(rs(TOTAL).PValue),\'C1Q_P\':float(rs(\'C1Q_mature\').PValue),\n  \'sparse_subsets_no_P\':True,\'density_min_n\':3,\'KDE_bandwidth\':.28,\'figure_sets\':len(MANIFEST),\n  \'input_sha256\':before}\n (ROOT/\'validation/plot_validation.json\').write_text(json.dumps(validation,indent=2),encoding=\'utf-8\')\n documentation(df,stats)\n versions={\'numpy\':np.__version__,\'pandas\':pd.__version__,\'scipy\':scipy.__version__,\'matplotlib\':matplotlib.__version__}\n (ROOT/\'requirements.txt\').write_text(\'\\n\'.join(f\'{k}=={v}\' for k,v in versions.items())+\'\\n\',encoding=\'utf-8\')\n (ROOT/\'validation/python_versions.json\').write_text(json.dumps({\'python\':platform.python_version(),**versions},indent=2),encoding=\'utf-8\')\n print(\'Saved five figure sets: one main, three subsets, and one supplemental triptych.\')\n\nif __name__==\'__main__\':main()\n',
'subset_umap': '"""Redraw three macrophage-subset expression panels with a darker color scale.\n\nRun: python code/redraw_supplementary_umap.py [package_directory]\nCoordinates, annotations, expression values, point counts and model results are\nread from the included verified tables. Only visual styling and labels change.\n"""\nfrom pathlib import Path\nimport sys,json,hashlib,platform,html\nimport numpy as np\nimport pandas as pd\nimport matplotlib\nmatplotlib.use(\'Agg\')\nimport matplotlib.pyplot as plt\nfrom matplotlib.colors import LinearSegmentedColormap,Normalize\nfrom matplotlib.ticker import MaxNLocator,FormatStrFormatter\n\nROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1]\nSCOPES=[\'C1Q_mature\',\'ACOD1_inflammatory_like\',\'NRG1_activated_like\']\nPOP={\'C1Q_mature\':\'C1Q+ mature macrophages\',\n \'ACOD1_inflammatory_like\':\'ACOD1+ inflammatory macrophage-like cells\',\n \'NRG1_activated_like\':\'NRG1+ activated macrophage-like cells\'}\nGROUPS=[\'Control\',\'UC\'];NAVY,WINE,INK=\'#173F63\',\'#AF1D40\',\'#202B34\'\nCMAP=LinearSegmentedColormap.from_list(\'TYROBP_deep_purple\',\n [(0,\'#D7CCD9\'),(.15,\'#D569AD\'),(.4,\'#B01A78\'),(.7,\'#75054F\'),(1,\'#330022\')])\nNORM=Normalize(0,5);MANIFEST=[];PLOTTED=[]\n\ndef theme():\n plt.rcParams.update({\'font.family\':\'Arial\',\'font.size\':10,\'axes.labelsize\':10.5,\n  \'axes.titlesize\':10.5,\'axes.edgecolor\':INK,\'axes.labelcolor\':INK,\'text.color\':INK,\n  \'xtick.color\':INK,\'ytick.color\':INK,\'axes.linewidth\':.9,\'pdf.fonttype\':42,\n  \'svg.fonttype\':\'none\',\'mathtext.fontset\':\'dejavusans\',\'figure.facecolor\':\'white\',\n  \'savefig.facecolor\':\'white\',\'xtick.major.size\':3,\'ytick.major.size\':3})\n\ndef clean(ax):\n ax.spines[[\'top\',\'right\']].set_visible(False);ax.grid(False)\n\ndef save(fig,folder,stem,description):\n out=ROOT/\'figures\'/folder;out.mkdir(parents=True,exist_ok=True)\n for ext in [\'png\',\'pdf\',\'svg\']:\n  fig.savefig(out/f\'{stem}.{ext}\',dpi=600,bbox_inches=\'tight\',pad_inches=.08)\n fig.savefig(out/f\'{stem}_preview.png\',dpi=150,bbox_inches=\'tight\',pad_inches=.08)\n MANIFEST.append(dict(folder=folder,stem=stem,description=description))\n plt.close(fig)\n\ndef feature(ax,cells,scope,group,limits):\n bg=cells[cells.group.eq(group)]\n target=bg[bg.population.eq(POP[scope])].sort_values(\'TYROBP_log_normalized\')\n ax.scatter(bg.UMAP_1,bg.UMAP_2,s=5,c=\'#DEE1E4\',alpha=.75,linewidths=0,zorder=1)\n ax.scatter(target.UMAP_1,target.UMAP_2,c=target.TYROBP_log_normalized,\n  cmap=CMAP,norm=NORM,s=23 if len(target)<=4 else 18,alpha=1,\n  edgecolor=\'#FFFFFF90\',linewidth=.16,zorder=2)\n ax.set_xlim(*limits[0]);ax.set_ylim(*limits[1]);ax.set_aspect(\'equal\')\n ax.set_title(f\'{group}  |  {len(target)} cells\',pad=8)\n ax.set_xticks([]);ax.set_yticks([]);ax.set_xlabel(\'UMAP 1\');ax.set_ylabel(\'UMAP 2\');clean(ax)\n if not len(target):ax.text(.5,.48,\'No observed cells\',transform=ax.transAxes,ha=\'center\',va=\'center\',fontsize=10,color=\'#58636C\')\n for r in target.itertuples():\n  PLOTTED.append(dict(scope=scope,group=group,cell=r.cell,UMAP_1=r.UMAP_1,UMAP_2=r.UMAP_2,\n                      TYROBP_log_normalized=r.TYROBP_log_normalized))\n\ndef expression_panel(ax,expr,scope,stat):\n d=expr[expr.scope.eq(scope)]\n arrays=[d.loc[d.group.eq(g),\'TMM_log2CPM\'].to_numpy() for g in GROUPS]\n joint=np.concatenate(arrays);span=max(np.ptp(joint),1.);low,high=joint.min()-.19*span,joint.max()+.36*span\n rng=np.random.default_rng(20261117+sum(map(ord,scope)))\n for i,(v,color) in enumerate(zip(arrays,[NAVY,WINE])):\n  if not len(v):\n   ax.text(i,low+.44*(high-low),\'No observed\\ncells\',ha=\'center\',va=\'center\',color=\'#58636C\',fontsize=9)\n   continue\n  if len(v)>=3:\n   ax.boxplot([v],positions=[i],widths=.40,patch_artist=True,showfliers=False,\n    boxprops=dict(facecolor=color+\'22\',edgecolor=color,linewidth=1.1,zorder=1),\n    medianprops=dict(color=INK,linewidth=1.6,zorder=2),\n    whiskerprops=dict(color=color,linewidth=1,zorder=1),capprops=dict(color=color,linewidth=1,zorder=1))\n  offsets=np.linspace(-.14,.14,len(v));rng.shuffle(offsets)\n  ax.scatter(i+offsets,v,s=33,c=color,edgecolor=\'white\',lw=.6,zorder=4)\n ax.set_xlim(-.57,1.57);ax.set_ylim(low,high);clean(ax)\n ax.set_xticks([0,1],[f\'{g}\\n(n = {len(v)})\' for g,v in zip(GROUPS,arrays)])\n ax.set_ylabel(\'TYROBP expression (log$_2$ CPM)\')\n ax.set_title(\'TYROBP expression\',pad=8)\n ax.yaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3));ax.yaxis.set_major_formatter(FormatStrFormatter(\'%.1f\'))\n if pd.notna(stat.PValue):\n  y=joint.max()+.18*span;h=.035*span\n  ax.plot([0,0,1,1],[y-h,y,y,y-h],lw=.8,c=INK)\n  ax.text(.5,y+.6*h,rf\'$P = {stat.PValue:.3f}$\',ha=\'center\',va=\'bottom\',fontsize=10)\n else:ax.text(.5,.975,\'Descriptive only\',transform=ax.transAxes,ha=\'center\',va=\'top\',fontsize=9,color=\'#58636C\')\n\ndef bar(fig,rect,vertical=False):\n cb=fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP,norm=NORM),cax=fig.add_axes(rect),\n                 orientation=\'vertical\' if vertical else \'horizontal\',ticks=[0,1,2,3,4,5])\n cb.outline.set_visible(False);cb.ax.tick_params(labelsize=9)\n cb.set_label(\'TYROBP expression (log-normalized)\',fontsize=10)\n\ndef docs():\n guide=\'\'\'TYROBP 亚群表达补充图：深紫红配色版（2026-11-17）\n\n适合放在补充材料，作为正文总巨噬细胞结果的亚群补充。\nUMAP 的有色点表示指定亚群中 TYROBP 的单细胞表达；不是 C1Q、ACOD1 或 NRG1 基因表达。\nC1Q、ACOD1、NRG1 是细胞群名称。浅灰背景是同组其他髓系细胞，仅用于定位。\nUMAP 使用原有全部髓系嵌入的坐标，未单独重新计算各亚群的 UMAP。\n深紫红色为更高的 TYROBP 表达。所有对照/UC面板使用共同的线性 0–5 色阶。\n表达值为 ln(1+10000×TYROBP_UMI/total_UMI)，自然对数归一化；本次只修改颜色、点大小及标签。\n全图顶部、右列标题和颜色标尺均标注 TYROBP expression。\n\n右列是供者层面的 pseudobulk 表达，TMM log2 CPM；每点一位实际检出该群细胞的供者。\n横轴只显示组名和 n；已去掉 donors 字样。n 不是细胞数，UMAP 标题中的 cells 才是细胞数。\nC1Q：Control 163 / UC 101 细胞、6/6 供者，P=0.846。\nACOD1：Control 0 / UC 160 细胞、0/6 供者；无对照亚群，不检验表达差异。\nNRG1：Control 4 / UC 168 细胞、2/3 供者；控制组覆盖过少，仅描述，不检验表达差异。\n缺失亚群不是零表达；不为缺失供者补零。\n图中 C1Q 的 P 来自既有 edgeR 准似然检验；BH FDR=0.967580，表格保留完整值。\nUMAP 表达颜色不能替代供者层面的统计检验，各图纵轴按本范围独立归一化，不作跨亚群直接比较。\n\n文件\nfigures/01_supplementary：三亚群的总补充图（UMAP+箱线散点）。\nfigures/02_individual_rows：每个亚群的 UMAP+箱线散点独立图。\nfigures/03_UMAP_only：每个亚群的 Control/UC UMAP 独立图。\n所有图提供 600 dpi PNG、矢量 PDF/SVG，以及预览 PNG。\ncode/redraw_supplementary_umap.py：可运行的绘图代码。\ndata：原始验证后的绘图表，含固定坐标、注释、表达值、供者汇总和统计。\nvalidation：颜色设置、数据不变核验、全部目标点绘图信息。\nFigure_legend_ZH_EN.txt：中英文补充图图注。\n运行：python code/redraw_supplementary_umap.py\n依赖：numpy、pandas、matplotlib。完整原始计数/统计模型仍保存于上一版 TYROBP_Macrophages_2026-11-17 包。\n\'\'\'\n legend=\'\'\'English\nTYROBP expression across macrophage subsets. (A) C1Q+ mature macrophages. (B) ACOD1+ inflammatory macrophage-like cells. (C) NRG1+ activated macrophage-like cells. Left and middle: Control and UC UMAPs colored by log-normalized TYROBP expression using a shared scale. Grey points indicate other myeloid cells. Right: donor-level pseudobulk expression; each point represents one donor with observed cells. P is from the edgeR quasi-likelihood test. Panels B and C are descriptive only owing to absent or sparse control populations.\n\n中文\n各巨噬细胞亚群中的 TYROBP 表达。（A）C1Q+ 成熟巨噬细胞；（B）ACOD1+ 炎症性巨噬细胞样细胞；（C）NRG1+ 活化巨噬细胞样细胞。左、中列为对照和 UC 的 UMAP，按对数归一化 TYROBP 表达着色，并使用共同色阶；灰色点表示其他髓系细胞。右列为供者层面的 pseudobulk 表达，每点代表一位检出相应细胞群的供者。P 值来自 edgeR 准似然检验。B、C 因缺少或仅有极少对照细胞，仅作描述性展示。\n\'\'\'\n (ROOT/\'README_图形说明.txt\').write_text(guide,encoding=\'utf-8-sig\')\n (ROOT/\'Figure_legend_ZH_EN.txt\').write_text(legend,encoding=\'utf-8-sig\')\n cards=[]\n for r in MANIFEST:\n  base=f\'figures/{r["folder"]}/{r["stem"]}\'\n  links=\' · \'.join(f\'<a href="{base}.{ext}">{ext.upper()}</a>\' for ext in [\'png\',\'pdf\',\'svg\'])\n  cards.append(f\'<article><h2>{html.escape(r["description"])}</h2><img src="{base}_preview.png"><p>{links}</p></article>\')\n page=\'\'\'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TYROBP 亚群表达补充图</title><style>body{margin:0;background:#f4f6f8;color:#243341;font:16px/1.7 Arial,"Microsoft YaHei",sans-serif}main{max-width:1120px;margin:auto;padding:30px}h1{font-size:27px}h2{font-size:19px}article{background:white;margin:24px 0;padding:20px;border-radius:7px}img{width:100%;height:auto}a{color:#174e79}.note{background:white;border-left:4px solid #B01A78;padding:18px}pre{white-space:pre-wrap;font:14px/1.7 Arial,"Microsoft YaHei",sans-serif;background:white;padding:20px}</style><main><h1>TYROBP 亚群表达补充图</h1><div class="note">UMAP 的颜色表示 TYROBP 表达强度，深紫红为高表达，浅灰为其他髓系背景。所有图使用共同的 0–5 线性色阶，坐标、数值和统计结果沿用已验证输入。</div><p><a href="Figure_legend_ZH_EN.txt">中英文图注</a> · <a href="README_图形说明.txt">作图说明</a></p>\'\'\'+\'\'.join(cards)+\'<pre>\'+html.escape(guide)+\'</pre></main></html>\'\n (ROOT/\'TYROBP_亚群补充图预览.html\').write_text(page,encoding=\'utf-8\')\n\ndef main():\n theme()\n paths=[ROOT/\'data\'/name for name in [\'TYROBP_cell_expression_and_UMAP.csv\',\'TYROBP_donor_expression.csv\',\'TYROBP_scope_results.csv\']]\n before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}\n cells=pd.read_csv(paths[0]);expr=pd.read_csv(paths[1]);stats=pd.read_csv(paths[2])\n assert len(cells)==1338 and cells.cell.is_unique\n assert cells.TYROBP_log_normalized.between(0,5).all()\n assert np.allclose(cells.TYROBP_log_normalized,np.log1p(1e4*cells.TYROBP_UMI/cells.total_UMI),atol=1e-12)\n limits=[]\n for col in [\'UMAP_1\',\'UMAP_2\']:\n  v=cells[col];pad=np.ptp(v)*.055;limits.append((v.min()-pad,v.max()+pad))\n def rs(scope):return stats[stats.scope.eq(scope)].iloc[0]\n # Three-row supplementary layout, matching the uploaded scope and box plots.\n fig=plt.figure(figsize=(11.6,11.15))\n gs=fig.add_gridspec(3,3,width_ratios=[1,1,1.10],wspace=.45,hspace=.65)\n fig.subplots_adjust(left=.075,right=.985,bottom=.14,top=.89)\n row_axes=[]\n for row,scope in enumerate(SCOPES):\n  axes=[fig.add_subplot(gs[row,col]) for col in range(3)]\n  for ax,g in zip(axes[:2],GROUPS):feature(ax,cells,scope,g,limits)\n  expression_panel(axes[2],expr,scope,rs(scope));row_axes.append(axes)\n fig.canvas.draw()\n for row,(scope,axes) in enumerate(zip(SCOPES,row_axes)):\n  y=max(ax.get_position().y1 for ax in axes)+.037\n  fig.text(.045,y,chr(65+row),fontsize=14,fontweight=\'bold\',ha=\'left\')\n  fig.text(.079,y,POP[scope],fontsize=11.5,ha=\'left\')\n fig.suptitle(\'TYROBP expression in macrophage subsets\',x=.51,y=.985,fontsize=15)\n bar(fig,[.16,.065,.40,.012])\n fig.text(.69,.062,\'Grey points: other myeloid cells\',fontsize=9,color=\'#58636C\',ha=\'left\')\n save(fig,\'01_supplementary\',\'Supplementary_TYROBP_subsets_UMAP_deep_purple\',\'补充图：三个亚群的 TYROBP 表达\')\n for row,scope in enumerate(SCOPES,1):\n  fig=plt.figure(figsize=(11.6,4.1));gs=fig.add_gridspec(1,3,width_ratios=[1,1,1.10],wspace=.45)\n  fig.subplots_adjust(left=.075,right=.985,bottom=.25,top=.78)\n  axes=[fig.add_subplot(gs[0,col]) for col in range(3)]\n  for ax,g in zip(axes[:2],GROUPS):feature(ax,cells,scope,g,limits)\n  expression_panel(axes[2],expr,scope,rs(scope))\n  fig.suptitle(POP[scope],x=.51,y=.975,fontsize=12)\n  bar(fig,[.16,.10,.40,.022])\n  save(fig,\'02_individual_rows\',f\'{row:02d}_{scope}_UMAP_and_expression\',\'独立面板：\'+POP[scope])\n  fig,axes=plt.subplots(1,2,figsize=(7.7,3.85),layout=\'constrained\')\n  for ax,g in zip(axes,GROUPS):feature(ax,cells,scope,g,limits)\n  fig.suptitle(POP[scope]+\'\\nTYROBP expression\',fontsize=11.5)\n  cb=fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP,norm=NORM),ax=list(axes),shrink=.70,pad=.035,fraction=.035,ticks=[0,1,2,3,4,5])\n  cb.outline.set_visible(False);cb.ax.tick_params(labelsize=9);cb.set_label(\'TYROBP expression\\n(log-normalized)\',fontsize=9)\n  save(fig,\'03_UMAP_only\',f\'{row:02d}_{scope}_TYROBP_UMAP\',\'UMAP：\'+POP[scope])\n plotted=pd.DataFrame(PLOTTED)\n assert len(plotted)==596*3\n original=cells.set_index(\'cell\')\n for cell,rows in plotted.groupby(\'cell\'):\n  r=original.loc[cell]\n  assert len(rows)==3 and (rows.UMAP_1==r.UMAP_1).all() and (rows.UMAP_2==r.UMAP_2).all()\n  assert (rows.TYROBP_log_normalized==r.TYROBP_log_normalized).all()\n for p in paths:assert before[p.name]==hashlib.sha256(p.read_bytes()).hexdigest()\n plotted.to_csv(ROOT/\'validation/plotted_feature_points.csv\',index=False)\n pd.DataFrame(MANIFEST).to_csv(ROOT/\'data/figure_manifest.csv\',index=False,encoding=\'utf-8-sig\')\n settings={\'color_name\':\'deep purple-magenta\',\'color_anchors\':[(0,\'#D7CCD9\'),(.15,\'#D569AD\'),(.4,\'#B01A78\'),(.7,\'#75054F\'),(1,\'#330022\')],\n  \'expression_color_scale\':[0,5],\'linear_mapping\':True,\'shared_between_all_panels\':True,\n  \'background\':\'Other myeloid cells, grey; not included in the target subset\',\n  \'plotted_target_cells\':596,\'feature_point_records\':len(plotted),\'input_tables_unchanged\':True,\n  \'target_coordinates_and_values_unchanged\':True,\'figure_sets\':len(MANIFEST),\n  \'axis_n_means\':\'Number of donors with observed subset cells; donors word omitted in figures\',\n  \'input_sha256\':before}\n (ROOT/\'validation/plot_validation.json\').write_text(json.dumps(settings,indent=2),encoding=\'utf-8\')\n versions={\'numpy\':np.__version__,\'pandas\':pd.__version__,\'matplotlib\':matplotlib.__version__}\n (ROOT/\'requirements.txt\').write_text(\'\\n\'.join(f\'{k}=={v}\' for k,v in versions.items())+\'\\n\',encoding=\'utf-8\')\n (ROOT/\'validation/python_versions.json\').write_text(json.dumps({\'python\':platform.python_version(),**versions},indent=2),encoding=\'utf-8\')\n docs();print(\'Saved seven figure sets with explicit TYROBP labels and shared deep-purple scale.\')\n\nif __name__==\'__main__\':main()\n',
'color_options': '"""Render three palette alternatives using exactly the same expression mapping.\n\nRun: python code/draw_color_options.py [package_directory]\n"""\nfrom pathlib import Path\nimport sys,json,hashlib,html\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom matplotlib.colors import LinearSegmentedColormap\nimport _plot_helpers as b\n\nROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1]\nb.ROOT=ROOT\nPALETTES={\n \'01_Orange_red\':dict(name=\'Orange-red\',Chinese=\'橙红\',anchors=[(0,\'#DFE3E6\'),(.2,\'#FDBB6C\'),(.4,\'#EF7D23\'),(.7,\'#C74913\'),(1,\'#7F2704\')]),\n \'02_Teal\':dict(name=\'Teal\',Chinese=\'蓝绿（推荐）\',anchors=[(0,\'#DFE3E6\'),(.2,\'#87C9C1\'),(.4,\'#279C90\'),(.7,\'#14797E\'),(1,\'#064550\')]),\n \'03_Deep_blue\':dict(name=\'Deep blue\',Chinese=\'深蓝\',anchors=[(0,\'#DFE3E6\'),(.2,\'#7CACCD\'),(.4,\'#337EAE\'),(.7,\'#17547C\'),(1,\'#09283E\')]),\n}\n\ndef cmap(key):return LinearSegmentedColormap.from_list(key,PALETTES[key][\'anchors\'])\n\ndef main():\n b.theme()\n paths=[ROOT/\'data\'/name for name in [\'TYROBP_cell_expression_and_UMAP.csv\',\'TYROBP_donor_expression.csv\',\'TYROBP_scope_results.csv\']]\n hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}\n cells=pd.read_csv(paths[0]);expr=pd.read_csv(paths[1]);stats=pd.read_csv(paths[2])\n limits=[]\n for col in [\'UMAP_1\',\'UMAP_2\']:\n  v=cells[col];pad=np.ptp(v)*.055;limits.append((v.min()-pad,v.max()+pad))\n def rs(s):return stats[stats.scope.eq(s)].iloc[0]\n for key,item in PALETTES.items():\n  b.CMAP=cmap(key)\n  fig=plt.figure(figsize=(11.6,11.15));gs=fig.add_gridspec(3,3,width_ratios=[1,1,1.10],wspace=.45,hspace=.65)\n  fig.subplots_adjust(left=.075,right=.985,bottom=.14,top=.89);row_axes=[]\n  for row,s in enumerate(b.SCOPES):\n   axes=[fig.add_subplot(gs[row,c]) for c in range(3)]\n   for ax,g in zip(axes[:2],b.GROUPS):b.feature(ax,cells,s,g,limits)\n   b.expression_panel(axes[2],expr,s,rs(s));row_axes.append(axes)\n  fig.canvas.draw()\n  for row,(s,axes) in enumerate(zip(b.SCOPES,row_axes)):\n   y=max(ax.get_position().y1 for ax in axes)+.037\n   fig.text(.045,y,chr(65+row),fontsize=14,fontweight=\'bold\',ha=\'left\')\n   fig.text(.079,y,b.POP[s],fontsize=11.5,ha=\'left\')\n  fig.suptitle(\'TYROBP expression in macrophage subsets\',x=.51,y=.985,fontsize=15)\n  b.bar(fig,[.16,.065,.40,.012]);fig.text(.69,.062,\'Grey points: other myeloid cells\',fontsize=9,color=\'#58636C\')\n  b.save(fig,key,f\'Supplementary_TYROBP_subsets_{key}\',f\'{item["Chinese"]}：完整补充图\')\n  for i,s in enumerate(b.SCOPES,1):\n   fig,axes=plt.subplots(1,2,figsize=(7.7,3.85),layout=\'constrained\')\n   for ax,g in zip(axes,b.GROUPS):b.feature(ax,cells,s,g,limits)\n   fig.suptitle(b.POP[s]+\'\\nTYROBP expression\',fontsize=11.5)\n   cb=fig.colorbar(plt.cm.ScalarMappable(cmap=b.CMAP,norm=b.NORM),ax=list(axes),shrink=.7,pad=.035,fraction=.035,ticks=[0,1,2,3,4,5])\n   cb.outline.set_visible(False);cb.ax.tick_params(labelsize=9);cb.set_label(\'TYROBP expression\\n(log-normalized)\',fontsize=9)\n   b.save(fig,f\'{key}/UMAP_only\',f\'{i:02d}_{s}_TYROBP_UMAP\',f\'{item["Chinese"]}：\'+b.POP[s])\n # A compact comparison uses the same UC C1Q cells, same coordinates, and scale.\n fig,axes=plt.subplots(1,3,figsize=(10.6,3.55))\n fig.subplots_adjust(left=.055,right=.985,bottom=.27,top=.78,wspace=.22)\n for i,(ax,(key,item)) in enumerate(zip(axes,PALETTES.items())):\n  b.CMAP=cmap(key);b.feature(ax,cells,\'C1Q_mature\',\'UC\',limits)\n  ax.set_title(item[\'name\'],pad=8,fontsize=12)\n  if i:ax.set_ylabel(\'\')\n  pos=ax.get_position();cb=fig.colorbar(plt.cm.ScalarMappable(cmap=b.CMAP,norm=b.NORM),\n      cax=fig.add_axes([pos.x0,.15,pos.width,.025]),orientation=\'horizontal\',ticks=[0,2.5,5])\n  cb.outline.set_visible(False);cb.ax.tick_params(labelsize=8)\n fig.suptitle(\'TYROBP expression: three color options\',y=.985,fontsize=13)\n fig.text(.5,.035,\'Same UC C1Q+ cells; shared expression range 0–5 (log-normalized)\',ha=\'center\',fontsize=9)\n b.save(fig,\'00_comparison\',\'TYROBP_UMAP_palette_comparison\',\'三种配色直接比较\')\n for p in paths:assert hashes[p.name]==hashlib.sha256(p.read_bytes()).hexdigest()\n plotted=pd.DataFrame(b.PLOTTED)\n original=cells.set_index(\'cell\')\n for col in [\'UMAP_1\',\'UMAP_2\',\'TYROBP_log_normalized\']:\n  assert np.array_equal(plotted[col].to_numpy(),original.loc[plotted.cell,col].to_numpy())\n plotted.to_csv(ROOT/\'validation/plotted_points.csv\',index=False)\n pd.DataFrame(b.MANIFEST).to_csv(ROOT/\'data/figure_manifest.csv\',index=False,encoding=\'utf-8-sig\')\n (ROOT/\'validation/color_options.json\').write_text(json.dumps({\'palettes\':PALETTES,\'shared_range\':[0,5],\n  \'same_linear_normalization\':True,\'all_target_values_and_coordinates_unchanged\':True,\n  \'plotted_feature_point_records\':len(plotted),\'figure_sets\':len(b.MANIFEST),\'input_sha256\':hashes},indent=2,ensure_ascii=False),encoding=\'utf-8\')\n guide=\'\'\'TYROBP UMAP 三种配色（2026-11-17）\n\n01_Orange_red：橙红色。02_Teal：蓝绿色（推荐）。03_Deep_blue：较深的蓝色。\n00_comparison：以同一批 UC C1Q+ 细胞直接比较三种颜色，帮助选择样式。\n每个配色目录都有完整的三亚群补充图和各亚群独立 UMAP。\n均提供 PNG（600 dpi）、PDF、SVG 和预览 PNG。\n\n有色点为当前指定亚群的 TYROBP 表达；浅灰点为其他髓系细胞的定位背景。\n全部配色使用同一个0–5线性表达标尺；n、P、UMAP坐标、点大小及数值均一致。\n三种颜色只用于选择图形样式，不代表三种生物学状态。\n右列为供者 pseudobulk 表达：C1Q n=6/6、P=.846；ACOD1 n=0/6、NRG1 n=2/3仅描述。\n不为缺失亚群补零表达。图中n仍为实际供者数，未显示donors字样。\n\n运行 python code/draw_color_options.py 可从随附 data/ 表复现全部图。\n依赖 numpy、pandas、matplotlib。详细输入指纹和逐点核验见 validation/。\n\'\'\'\n (ROOT/\'README_三种配色说明.txt\').write_text(guide,encoding=\'utf-8-sig\')\n cards=[]\n for key,item in PALETTES.items():\n  base=f\'figures/{key}/Supplementary_TYROBP_subsets_{key}\'\n  cards.append(f\'<article><h2>{item["Chinese"]}</h2><img src="{base}_preview.png"><p>\'+ \' · \'.join(f\'<a href="{base}.{e}">{e.upper()}</a>\' for e in [\'png\',\'pdf\',\'svg\'])+\'</p></article>\')\n page=\'\'\'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TYROBP UMAP 配色选择</title><style>body{font:16px/1.7 Arial,"Microsoft YaHei",sans-serif;background:#f4f6f8;color:#243341;margin:0}main{max-width:1100px;margin:auto;padding:30px}h1{font-size:27px}h2{font-size:20px}img{width:100%;height:auto}article{background:white;padding:20px;margin:24px 0;border-radius:8px}a{color:#17547c}pre{white-space:pre-wrap;background:white;padding:20px}</style><main><h1>TYROBP UMAP 配色选择</h1><p>左至右：橙红、蓝绿、深蓝。我更推荐蓝绿色。三种方案的数据、坐标和表达色阶相同。</p><img src="figures/00_comparison/TYROBP_UMAP_palette_comparison_preview.png">\'\'\'+\'\'.join(cards)+\'<pre>\'+html.escape(guide)+\'</pre></main></html>\'\n (ROOT/\'TYROBP_UMAP_配色选择.html\').write_text(page,encoding=\'utf-8\')\n (ROOT/\'requirements.txt\').write_text(f\'numpy=={np.__version__}\\npandas=={pd.__version__}\\nmatplotlib=={b.matplotlib.__version__}\\n\',encoding=\'utf-8\')\n print(\'Saved 3 full supplementary palettes, 9 standalone UMAPs, and 1 comparison.\')\n\nif __name__==\'__main__\':main()\n',
'donor_separate': r'''"""Draw two stand-alone donor figures using the previously verified GSE214695 analysis."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DONORS = [f"{g}{n}" for g in ("HC", "UC") for n in range(1, 7)]
X = np.r_[np.arange(6), np.arange(6,12) + 0.68]
XLIM = (-0.55, X[-1] + 0.55)
MAJOR = ["B_Plasma", "T_cells", "Stromal", "Epithelial", "Myeloid"]
SUBSETS = ["C1Q_mature", "ACOD1_inflammatory_like", "NRG1_activated_like"]
POOL = "Combined_macrophage_like"
ALL = "All_captured"
POPULATIONS = [ALL, "Myeloid", POOL, *SUBSETS]
MAJOR_COLORS = ["#748FA6", "#B8CED7", "#B6B498", "#DDC5A9", "#405F6A"]
SUBSET_COLORS = ["#407F81", "#B24862", "#C79245"]
BLUE, WINE = "#1D4564", "#982D4C"
INK, MUTED = "#253139", "#4F6068"
GRID, RULE, MISSING = "#DEE5E7", "#ABB8BC", "#E8ECEE"
SITE_MARKERS = {"Sigmoid colon":"o", "Rectum":"^", "Transverse colon":"D"}

def setup_font():
    for path in Path("C:/Windows/Fonts").glob("times*.ttf"):
        font_manager.fontManager.addfont(str(path))
    family = "Times New Roman"
    assert Path(font_manager.findfont(font_manager.FontProperties(family=family),fallback_to_default=False)).is_file()
    plt.rcParams.update({
        "font.family":family,"font.weight":"normal","axes.titleweight":"normal",
        "font.size":10,"axes.labelcolor":INK,"text.color":INK,"axes.edgecolor":"#65777E",
        "xtick.color":MUTED,"ytick.color":MUTED,"axes.linewidth":.65,"axes.unicode_minus":False,
        "pdf.fonttype":42,"ps.fonttype":42,"mathtext.fontset":"custom",
        "mathtext.rm":family,"mathtext.it":family+":italic","mathtext.bf":family+":bold",
        "figure.facecolor":"white","savefig.facecolor":"white"})

def read_results():
    source=BASE/"results"
    counts=pd.read_csv(source/"donor_counts.csv").set_index("donor").loc[DONORS]
    assert counts.index.is_unique and counts.GSM.is_unique and len(counts)==12
    assert counts.group.tolist()==["Control"]*6+["UC"]*6
    assert (counts[MAJOR].sum(axis=1)==counts[ALL]).all()
    assert (counts[SUBSETS].sum(axis=1)==counts[POOL]).all()
    assert counts[ALL].sum()==21965 and counts[POOL].sum()==596
    expr=pd.read_csv(source/"donor_expression_complete.csv").query("min_cells == 1")
    expr=expr.set_index(["scope","donor"])
    assert expr.index.is_unique
    for scope in POPULATIONS:
        rows=expr.loc[scope].loc[DONORS]
        assert rows.captured_cells.equals(counts[scope].rename("captured_cells"))
        assert (rows.log2CPM.notna()==(rows.captured_cells>0)).all()
        assert np.array_equal(rows.site,counts.site) and np.array_equal(rows.GSM,counts.GSM)
    return counts,expr

def donor_color(index):
    return BLUE if index<6 else WINE

def xfig(x,left,width):
    return left+width*(x-XLIM[0])/(XLIM[1]-XLIM[0])

def group_headings(fig,left,width,y):
    for indices,label,color in [(X[:6],"Healthy controls",BLUE),(X[6:],"Ulcerative colitis",WINE)]:
        first=xfig(indices[0]-.38,left,width)
        last=xfig(indices[-1]+.38,left,width)
        fig.text((first+last)/2,y,label,ha="center",va="bottom",fontsize=10.3,color=color)
        fig.add_artist(Line2D([first,last],[y-.005,y-.005],transform=fig.transFigure,
                              color=color,alpha=.60,lw=.65))

def aligned_row(fig,left,width,y,values,label=None,donor_colors=False,size=9.1):
    if label:
        fig.text(left-.018,y,label,ha="right",va="center",fontsize=8.3,color=MUTED)
    for i,s in enumerate(values):
        fig.text(xfig(X[i],left,width),y,s,ha="center",va="center",fontsize=size,
                 color=donor_color(i) if donor_colors else MUTED)

def annotations(fig,x,y,letter,heading):
    fig.text(.040,y,letter,fontsize=17.8,va="center")
    fig.text(x,y,heading,fontsize=14.4,va="center")

def stacked_panel(fig,counts,left,width,bottom,height,keys,denominator,palette,ylabel):
    ax=fig.add_axes([left,bottom,width,height])
    base=np.zeros(len(DONORS))
    for key,color in zip(keys,palette):
        heights=100*(counts[key]/counts[denominator]).values
        ax.bar(X,heights,bottom=base,width=.77,color=color,edgecolor="white",linewidth=.32,zorder=3)
        base+=heights
    assert np.allclose(base,100)
    ax.set_xlim(*XLIM);ax.set_ylim(0,100)
    ax.set_yticks([0,25,50,75,100])
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(100,decimals=0))
    ax.set_xticks(X,DONORS)
    ax.tick_params(axis="y",length=3,width=.65,labelsize=9.3,pad=4)
    ax.tick_params(axis="x",length=0,labelsize=10,pad=4.8)
    for i,t in enumerate(ax.get_xticklabels()): t.set_color(donor_color(i))
    ax.set_ylabel(ylabel,labelpad=6.0,fontsize=10.3)
    ax.spines[["right","top"]].set_visible(False)
    ax.set_axisbelow(True);ax.yaxis.grid(True,color=GRID,lw=.57)
    ax.axvline((X[5]+X[6])/2,color=RULE,ls=(0,(2,3)),lw=.62,zorder=2)
    return ax

def patches_legend(fig,x,y,names,colors):
    handles=[Patch(facecolor=color,edgecolor="none",label=label) for color,label in zip(colors,names)]
    fig.legend(handles=handles,loc="center left",bbox_to_anchor=(x,y),ncol=len(names),
               frameon=False,borderaxespad=0,handlelength=1.00,handleheight=.78,
               handletextpad=.4,columnspacing=1.08,fontsize=10)

def proportion_figure(counts):
    fig=plt.figure(figsize=(8.10,6.60))
    left,width=.137,.834
    annotations(fig,.089,.954,"A","Major cell populations by donor")
    patches_legend(fig,left,.903,["B and plasma cells","T cells","Stromal cells","Epithelial cells","Myeloid cells"],MAJOR_COLORS)
    group_headings(fig,left,width,.841)
    a=stacked_panel(fig,counts,left,width,.586,.239,MAJOR,ALL,MAJOR_COLORS,"Share of all recovered cells")
    aligned_row(fig,left,width,.523,[f"{v:,}" for v in counts[ALL]],"All cells (n)")
    annotations(fig,.089,.449,"B","Composition of macrophage-related cells by donor")
    patches_legend(fig,left,.394,["C1Q⁺ mature","ACOD1⁺ inflammatory-like","NRG1⁺ activated-like"],SUBSET_COLORS)
    b=stacked_panel(fig,counts,left,width,.135,.227,SUBSETS,POOL,SUBSET_COLORS,"Share within the three subsets")
    aligned_row(fig,left,width,.079,[str(n) for n in counts[POOL]],"Pooled cells (n)")
    aligned_row(fig,left,width,.045,[f"{100*n/m:.2f}%" for n,m in zip(counts[POOL],counts[ALL])],"Pool / all cells")
    assert len(a.patches)==5*12 and len(b.patches)==3*12
    return fig

def marker_key(label,marker,filled=True):
    return Line2D([],[],marker=marker,linestyle="none",markersize=5.1,
                  markeredgewidth=.9,markerfacecolor=MUTED if filled else "white",
                  markeredgecolor=MUTED,label=label)

def expression_legend(fig,x,y,text,handles):
    fig.text(.057,y,text,fontsize=9.1,color=MUTED,va="center")
    fig.legend(handles=handles,loc="center left",bbox_to_anchor=(x,y),frameon=False,
               ncol=len(handles),fontsize=9.5,handlelength=1.15,handletextpad=.40,
               columnspacing=1.65,borderaxespad=0)

def expression_figure(counts,expr):
    fig=plt.figure(figsize=(8.10,7.72))
    fig.text(.043,.962,r"$\mathit{TYROBP}$ expression by donor",fontsize=17.4,va="center")
    fig.text(.045,.924,"Donor pseudobulk expression (log₂ CPM), TMM-normalized within each cell population",
             va="center",fontsize=10.3,color=MUTED)
    labels=["All recovered cells","All myeloid cells","Macrophage-related\ncells pooled",
            "C1Q⁺ mature\nmacrophages","ACOD1⁺ inflammatory\nmacrophage-like cells","NRG1⁺ activated\nmacrophage-like cells"]
    left,width=.289,.682
    group_headings(fig,left,width,.876)
    aligned_row(fig,left,width,.846,DONORS,donor_colors=True,size=9.6)
    plotted=0
    for j,(scope,label) in enumerate(zip(POPULATIONS,labels)):
        height=.087
        top=.823-j*.114
        ax=fig.add_axes([left,top-height,width,height])
        ax.set_xlim(*XLIM)
        limits,yticks=((3.68,6.94),[4,5,6]) if scope==ALL else ((9.30,12.33),[10,11,12])
        ax.set_ylim(*limits);ax.set_yticks(yticks)
        ax.set_xticks(X,[]);ax.tick_params(axis="x",length=0)
        ax.tick_params(axis="y",length=2.5,pad=3.5,width=.55,labelsize=9)
        ax.spines[["right","top"]].set_visible(False)
        ax.spines["bottom"].set_linewidth(.5);ax.spines["bottom"].set_color("#C6CFD2")
        ax.spines["left"].set_linewidth(.55);ax.spines["left"].set_color("#96A5AA")
        ax.set_axisbelow(True)
        ax.yaxis.grid(True,color=GRID,lw=.52)
        ax.xaxis.grid(True,color="#EFF2F3",lw=.49)
        ax.axvline((X[5]+X[6])/2,color=RULE,ls=(0,(2,3)),lw=.6,zorder=1)
        frame=expr.loc[scope].loc[DONORS]
        for i,d in enumerate(DONORS):
            n=frame.at[d,"captured_cells"]
            if n==0:
                ax.axvspan(X[i]-.42,X[i]+.42,ymin=0,ymax=1,color=MISSING,lw=0,zorder=0.3)
                continue
            c=donor_color(i)
            ax.scatter(X[i],frame.at[d,"log2CPM"],s=41 if counts.at[d,"site"]!="Rectum" else 47,
                       marker=SITE_MARKERS[counts.at[d,"site"]],facecolor=c if n>=10 else "white",
                       edgecolor="white" if n>=10 else c,linewidth=.63 if n>=10 else 1.25,zorder=4)
            plotted+=1
        fig.text(.041,top-height*.46,chr(ord("A")+j),fontsize=13.4,va="center")
        fig.text(.076,top-height*.47,label,fontsize=10.5,va="center",linespacing=1.12)
    assert plotted==int(expr.loc[POPULATIONS,"log2CPM"].notna().sum())==59
    aligned_row(fig,left,width,.145,DONORS,donor_colors=True,size=9.6)
    expression_legend(fig,.283,.091,"Biopsy site",[marker_key(k,m) for k,m in SITE_MARKERS.items()])
    expression_legend(fig,.283,.053,"Cells in donor pseudobulk",[
        marker_key("≥10 cells","o"),marker_key("1–9 cells","o",False),
        Patch(facecolor=MISSING,edgecolor="#BAC5C9",linewidth=.6,label="No cells recovered")])
    return fig

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    setup_font()
    counts,expr=read_results()
    for name,figure in [("01_Donor_cell_proportions",proportion_figure(counts)),
                        ("02_Donor_TYROBP_expression",expression_figure(counts,expr))]:
        for suffix in ["pdf","png"]:
            figure.savefig(args.out/f"{name}.{suffix}",dpi=600,facecolor="white",
                           metadata={"Creator":"GSE214695 donor-level analysis"})
        plt.close(figure)
    print("12 donors; 21,965 recovered cells; 596 pooled macrophage-related cells; 59 measured expression points.")

if __name__=="__main__":
    main()
''',
'donor_alternatives': r'''"""Two views of the same single-donor pseudobulks: grouped distributions and a donor heatmap."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

from plot_separate_donor_figures import (
    ALL, BLUE, DONORS, INK, MUTED, POPULATIONS, WINE, read_results, setup_font,
)

NAMES = [
    "All cells after quality control", "All myeloid cells", "Macrophage-related\ncells pooled",
    "C1Q⁺ mature macrophages", "ACOD1⁺ inflammatory\nmacrophage-like cells",
    "NRG1⁺ activated\nmacrophage-like cells",
]
UNOBSERVED = "#E9ECEE"
RULE = "#CFD8DB"
MEDIAN_COLOR = "#526369"
SIZE = 119
SWARM_DIAMETER_PT = 12

def beeswarm_x(ax, center, values):
    """Minimum horizontal shifts for legible donor IDs, preserving exact vertical values."""
    screen = ax.transData.transform(np.column_stack([np.full(len(values), center), values]))
    diameter = SWARM_DIAMETER_PT * ax.figure.dpi / 72
    offsets = np.zeros(len(values))
    placed = []
    for idx in np.argsort(values, kind="stable"):
        y = screen[idx,1]
        candidates = [0.]
        for other in placed:
            gap = abs(y - screen[other,1])
            if gap < diameter:
                dx = np.sqrt(diameter**2 - gap**2) + .025
                candidates.extend([offsets[other]-dx, offsets[other]+dx])
        # Equal-distance solutions alternate sides of the center.
        direction = 1 if len(placed)%2 else -1
        candidates.sort(key=lambda v:(round(abs(v),5),direction*v))
        offsets[idx] = next(v for v in candidates if all(
            np.hypot(v-offsets[k],y-screen[k,1]) >= diameter-1e-4 for k in placed))
        placed.append(idx)
    screen[:,0] += offsets
    result = ax.transData.inverted().transform(screen)[:,0]
    assert np.max(np.abs(result-center),initial=0)<.44
    return result

def group_plot(expr):
    fig=plt.figure(figsize=(8.65,5.92))
    fig.text(.047,.957,r"$\mathit{TYROBP}$ expression in healthy controls and UC",
             fontsize=17.1,va="center")
    fig.text(.049,.907,
             "Each point represents one donor. Numbers inside points identify donors 1–6 in each group.",
             fontsize=10.5,color=MUTED,va="center")
    fig.text(.018,.514,"Donor pseudobulk expression (log₂ CPM)",rotation=90,
             va="center",ha="center",fontsize=10.6)
    plotted={}
    for j,scope in enumerate(POPULATIONS):
        left=[.091,.407,.723][j%3]
        bottom=.557 if j<3 else .178
        height=.235
        ax=fig.add_axes([left,bottom,.239,height])
        ax.set_xlim(-.56,1.56)
        ax.set_ylim(*( (3.52,7.12) if scope==ALL else (9.10,12.50) ))
        ax.set_yticks([4,5,6,7] if scope==ALL else [10,11,12])
        ax.tick_params(axis="y",labelsize=9.4,pad=3.3,width=.6,length=2.8)
        ax.tick_params(axis="x",labelsize=9.7,pad=5,width=0,length=0)
        ax.spines[["top","right"]].set_visible(False)
        ax.spines["left"].set_color("#86989E")
        ax.spines["bottom"].set_color("#B2BEC2")
        ax.spines["bottom"].set_linewidth(.6)
        ax.set_axisbelow(True)
        ax.yaxis.grid(color="#E3E8E9",linewidth=.52)
        fig.text(left-.047,bottom+height+.051,chr(65+j),fontsize=15.2,va="center")
        fig.text(left-.011,bottom+height+.051,NAMES[j],fontsize=12.0,va="center",linespacing=1.03)
        cells=expr.loc[scope].loc[DONORS]
        group_labels=[]
        for center,ids,color,tag in [(0,DONORS[:6],BLUE,"HC"),(1,DONORS[6:],WINE,"UC")]:
            measured=cells.loc[ids].dropna(subset=["log2CPM"])
            group_labels.append(f"{tag} (n = {len(measured)})")
            if measured.empty:
                ax.text(center,.53,"No cells\nrecovered",transform=ax.get_xaxis_transform(),
                        fontsize=9.3,ha="center",va="center",color="#75858B",linespacing=1.08,
                        bbox={"facecolor":"white","edgecolor":"none","pad":1.2})
                continue
            values=measured.log2CPM.to_numpy()
            xs=beeswarm_x(ax,center,values)
            if len(measured)>=3:
                ax.plot([center-.32,center+.32],[np.median(values)]*2,lw=1.15,
                        color=MEDIAN_COLOR,solid_capstyle="round",zorder=2)
            for (donor,row),x,y in zip(measured.iterrows(),xs,values):
                few=row.captured_cells<10
                marker=ax.scatter(x,y,s=SIZE,marker="o",facecolor="white" if few else color,
                                  edgecolor=color if few else "white",
                                  linewidth=1.05 if few else .6,zorder=3)
                marker.set_gid(scope+"_"+donor)
                digit=ax.text(x,y,donor[-1],ha="center",va="center",fontsize=7.0,
                              color=color if few else "white",zorder=4)
                digit.set_gid("id_"+scope+"_"+donor)
                plotted[(scope,donor)]=y
        ax.set_xticks([0,1],group_labels)
        for tick,color in zip(ax.get_xticklabels(),[BLUE,WINE]):tick.set_color(color)
    handles=[
        Line2D([],[],marker="o",color="none",markerfacecolor=MEDIAN_COLOR,
               markeredgecolor=MEDIAN_COLOR,markersize=6,label="≥10 cells"),
        Line2D([],[],marker="o",color="none",markerfacecolor="white",
               markeredgecolor=MEDIAN_COLOR,markersize=6,label="1–9 cells"),
        Line2D([],[],color=MEDIAN_COLOR,lw=1.15,label="Median (≥3 donors)"),
    ]
    fig.legend(handles=handles,loc="center left",bbox_to_anchor=(.050,.087),ncol=3,
               borderaxespad=0,frameon=False,fontsize=10.0,columnspacing=2,
               handlelength=1.5,handletextpad=.47)
    fig.text(.050,.043,"n gives the number of donors with recovered cells. Expression was normalized within each population.",
             fontsize=9.3,color=MUTED,va="center")
    assert len(plotted)==59
    return fig

def text_ink(face):
    rgb=np.array(to_rgb(face))
    linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    luminance=float(linear@np.array([.2126,.7152,.0722]))
    return "white" if luminance<.22 else "#1F2C30"

def heatmap(counts,expr):
    fig=plt.figure(figsize=(8.65,4.75))
    fig.text(.046,.944,r"$\mathit{TYROBP}$ expression by donor and cell population",
             fontsize=16.8,va="center")
    fig.text(.048,.876,"Values show donor pseudobulk expression (log₂ CPM).",
             fontsize=10.6,color=MUTED,va="center")
    left,width,bottom,height=.260,.712,.260,.431
    ax=fig.add_axes([left,bottom,width,height])
    xs=np.r_[np.arange(6),np.arange(6,12)+.34]
    ys=np.r_[np.arange(3),np.arange(3,6)+.18]
    xlim=(-.5,xs[-1]+.5)
    ax.set_xlim(*xlim);ax.set_ylim(ys[-1]+.5,-.5);ax.set_axis_off()
    def page_x(v):
        return left+width*(v-xlim[0])/(xlim[1]-xlim[0])
    for cols,label,color in [(xs[:6],"Healthy controls",BLUE),(xs[6:],"Ulcerative colitis",WINE)]:
        first,last=page_x(cols[0]-.48),page_x(cols[-1]+.48)
        fig.text((first+last)/2,.793,label,color=color,ha="center",va="center",fontsize=12.0)
        fig.add_artist(Line2D([first,last],[.768,.768],transform=fig.transFigure,color=color,lw=.85))
    for d,x in zip(DONORS,xs):
        fig.text(page_x(x),.732,d,ha="center",va="center",fontsize=10.7,
                 color=BLUE if d.startswith("HC") else WINE)

    cmap=LinearSegmentedColormap.from_list("teal_ivory_copper",[
        (0,"#216465"),(.25,"#8ABDB3"),(.5,"#FAF8F1"),(.75,"#DFAB82"),(1,"#9F5137")],N=256)
    norm=Normalize(-2,2,clip=False)
    offsets=[]
    for j,(scope,y) in enumerate(zip(POPULATIONS,ys)):
        row=expr.loc[scope].loc[DONORS]
        median=row.log2CPM.median()
        ax.text(-.71,y,NAMES[j],ha="right",va="center",fontsize=10.25,
                linespacing=1.03,clip_on=False)
        for d,x in zip(DONORS,xs):
            v=row.at[d,"log2CPM"];n=row.at[d,"captured_cells"]
            missing=np.isnan(v)
            if missing:
                face=UNOBSERVED; label="–"
            else:
                deviation=float(v-median);offsets.append(deviation)
                face=cmap(norm(deviation))
                label=f"{v:.2f}"+("†" if n<10 else "")
            cell=Rectangle((x-.487,y-.477),.974,.954,facecolor=face,
                           edgecolor="#E4E7E5" if not missing else "#E0E4E5",linewidth=.25)
            cell.set_gid(scope+"_"+d)
            ax.add_patch(cell)
            txt=ax.text(x,y,label,ha="center",va="center",fontsize=9.85,
                        color="#73828B" if missing else text_ink(face))
            txt.set_gid("value_"+scope+"_"+d)
    assert min(offsets)>=-2 and max(offsets)<=2 and len(offsets)==59
    sites={"Sigmoid colon":"Sigmoid","Rectum":"Rectum","Transverse colon":"Transverse"}
    row_y=.223
    fig.text(left-.014,row_y,"Biopsy site",ha="right",va="center",fontsize=9.3,color=MUTED)
    for x,d in zip(xs,DONORS):
        fig.text(page_x(x),row_y,sites[counts.at[d,"site"]],ha="center",
                 va="center",fontsize=8.3,color=MUTED)
    fig.text(.047,.139,"Colour shows log₂ CPM relative to the median of donors with cells in each row.",
             fontsize=9.5,va="center",color=MUTED)
    bar_ax=fig.add_axes([.049,.071,.426,.031])
    mesh=np.linspace(-2,2,256)[None,:]
    bar_ax.imshow(mesh,extent=(-2,2,0,1),aspect="auto",cmap=cmap,norm=norm,
                  interpolation="nearest",rasterized=True)
    bar_ax.set_yticks([])
    bar_ax.set_xticks([-2,-1,0,1,2],["−2","−1","0","+1","+2"])
    bar_ax.tick_params(axis="x",labelsize=8.5,length=2,pad=2,width=.55)
    bar_ax.spines[:].set_visible(False)
    fig.text(.57,.132,"†  Based on 1–9 cells",fontsize=9.7,color=INK,va="center")
    fig.add_artist(Rectangle((.57,.061),.022,.034,transform=fig.transFigure,
                             facecolor=UNOBSERVED,edgecolor=RULE,linewidth=.55))
    fig.text(.581,.078,"–",ha="center",va="center",fontsize=9.5,color=MUTED)
    fig.text(.601,.078,"No cells of this type after quality control",fontsize=9.7,color=INK,va="center")
    return fig

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    setup_font()
    counts,expr=read_results()
    for stem,fig in [("01_TYROBP_grouped_donor_plot",group_plot(expr)),
                     ("02_TYROBP_donor_heatmap",heatmap(counts,expr))]:
        for ext in ("png","pdf"):
            fig.savefig(args.out/(stem+"."+ext),dpi=600,facecolor="white",
                        metadata={"Creator":"GSE214695 donor analysis"})
        plt.close(fig)

if __name__=="__main__":
    main()
''',
'donor_combined': r'''"""Plot GSE214695 cell fractions and TYROBP pseudobulk values on one donor axis."""
import argparse
from pathlib import Path

from plot_separate_donor_figures import (
    ALL, BLUE, DONORS, MAJOR, MUTED, POOL, POPULATIONS,
    SUBSETS, SUBSET_COLORS, WINE, X, XLIM, aligned_row, donor_color,
    read_results, setup_font, stacked_panel, xfig,
)
from plot_expression_alternatives import NAMES, UNOBSERVED, text_ink
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np


WIDTH, HEIGHT = 7.25, 9.16
LEFT, PLOT_WIDTH = .242, .730
MAJOR_COLORS = ['#31577F', '#328EAB', '#687945', '#C47735', '#573650']
STEM = 'GSE214695_donor_composition_and_TYROBP'


def from_top(inches):
    return 1 - inches / HEIGHT


def panel_title(fig, depth, letter, title):
    y = from_top(depth)
    fig.text(.031, y, letter, va='center', fontsize=15.0)
    fig.text(.078, y, title, va='center', fontsize=12.8)


def population_key(fig, depth, names, colors):
    handles = [Patch(facecolor=color, edgecolor='none', label=name)
               for name, color in zip(names, colors)]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(LEFT + PLOT_WIDTH / 2, from_top(depth)),
               ncol=len(names), frameon=False, borderaxespad=0, fontsize=8.5,
               handlelength=1.0, handleheight=.78, handletextpad=.38, columnspacing=1.0)


def group_headings(fig, depth):
    for donors, name, color in [(X[:6], 'Healthy controls', BLUE), (X[6:], 'Ulcerative colitis', WINE)]:
        start = xfig(donors[0] - .48, LEFT, PLOT_WIDTH)
        end = xfig(donors[-1] + .48, LEFT, PLOT_WIDTH)
        fig.text((start + end) / 2, from_top(depth), name, ha='center', va='center',
                 fontsize=9.3, color=color)
        rule_y = from_top(depth + .107)
        fig.add_artist(Line2D([start, end], [rule_y, rule_y], transform=fig.transFigure,
                              color=color, linewidth=.6, alpha=.72))


def draw_bars(fig, counts, top, bottom, keys, denominator, colors, ylabel):
    ax = stacked_panel(fig, counts, LEFT, PLOT_WIDTH, from_top(bottom), (bottom - top) / HEIGHT,
                       keys, denominator, colors, ylabel)
    ax.tick_params(axis='x', labelsize=8.6, pad=3.2)
    ax.tick_params(axis='y', labelsize=8.2, pad=3.3)
    ax.yaxis.label.set_size(8.7)
    for key, container in zip(keys, ax.containers):
        for donor, bar in zip(DONORS, container.patches):
            bar.set_gid(f'bar_{key}_{donor}')
    return ax


def draw_expression(fig, counts, expr):
    top, bottom = 6.26, 7.99
    ax = fig.add_axes([LEFT, from_top(bottom), PLOT_WIDTH, (bottom - top) / HEIGHT])
    ys = np.r_[np.arange(3), np.arange(3, 6) + .20]
    ax.set_xlim(*XLIM)
    ax.set_ylim(ys[-1] + .5, -.5)
    ax.set_axis_off()
    cmap = LinearSegmentedColormap.from_list('expression_deviation',
           ['#216465', '#8ABDB3', '#FAF8F1', '#DFAB82', '#9F5137'], N=256)
    norm = Normalize(-2, 2)
    for scope, label, y in zip(POPULATIONS, NAMES, ys):
        values = expr.loc[scope].loc[DONORS]
        median = values.log2CPM.median()
        ax.text(XLIM[0] - .16, y, label, ha='right', va='center', fontsize=8.5,
                linespacing=1.04, clip_on=False)
        for x, donor in zip(X, DONORS):
            n = int(values.at[donor, 'captured_cells'])
            value = values.at[donor, 'log2CPM']
            if n:
                deviation = value - median
                assert norm.vmin <= deviation <= norm.vmax
                fill = cmap(norm(deviation))
                ink = text_ink(fill)
                number = f'{value:.2f}' + ('†' if n < 10 else '')
            else:
                fill, ink, number = UNOBSERVED, '#73828B', '–'
            tile = Rectangle((x - .485, y - .475), .97, .95, facecolor=fill,
                             edgecolor='#E1E6E6', linewidth=.28)
            ax.add_patch(tile)
            tile.set_gid(f'tile_{scope}_{donor}')
            t = ax.text(x, y, number, ha='center', va='center', fontsize=8.3, color=ink)
            t.set_gid(f'value_{scope}_{donor}')
    site_names = {'Sigmoid colon': 'Sigmoid', 'Transverse colon': 'Transverse', 'Rectum': 'Rectum'}
    aligned_row(fig, LEFT, PLOT_WIDTH, from_top(8.17), [site_names[s] for s in counts.site],
                'Biopsy site', size=7.0)
    fig.text(.041, from_top(8.54), 'Colour is centred on the median of donors with cells in each row.',
             fontsize=8.1, color=MUTED, va='center')
    scale = fig.add_axes([.041, from_top(8.80), .443, .10 / HEIGHT])
    gradient_x, gradient_y = np.meshgrid(np.linspace(-2, 2, cmap.N), [0, 1])
    scale.pcolormesh(gradient_x, gradient_y, gradient_x, cmap=cmap, norm=norm, shading='gouraud')
    scale.set_xlim(-2, 2)
    scale.set_yticks([])
    scale.set_xticks([-2, -1, 0, 1, 2], ['−2', '−1', '0', '+1', '+2'])
    scale.tick_params(axis='x', labelsize=7.5, length=2, width=.5, pad=2.2)
    scale.spines[:].set_visible(False)
    key_x = .548
    fig.text(key_x + .014, from_top(8.54), '†', fontsize=9, ha='center', va='center')
    fig.text(key_x + .039, from_top(8.54), 'Based on 1–9 cells', fontsize=8.4, va='center')
    key_y = from_top(8.78)
    fig.add_artist(Rectangle((key_x, key_y - .009), .028, .018, transform=fig.transFigure,
                             facecolor=UNOBSERVED, edgecolor='#C5CFD2', linewidth=.5))
    fig.text(key_x + .014, key_y, '–', fontsize=8.5, color=MUTED, ha='center', va='center')
    fig.text(key_x + .039, key_y, 'No cells of this type after quality control', fontsize=8.4, va='center')
    return ax


def combined_figure(counts, expr):
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    panel_title(fig, .21, 'A', 'Major cell populations by donor')
    population_key(fig, .47, ['B and plasma cells', 'T cells', 'Stromal cells', 'Epithelial cells',
                              'Myeloid cells'], MAJOR_COLORS)
    group_headings(fig, .69)
    draw_bars(fig, counts, .86, 2.04, MAJOR, ALL, MAJOR_COLORS, 'Share of all QC cells')
    aligned_row(fig, LEFT, PLOT_WIDTH, from_top(2.38), [f'{n:,}' for n in counts[ALL]],
                'All cells after QC (n)', size=8.1)

    panel_title(fig, 2.75, 'B', 'Composition of macrophage-related cells by donor')
    population_key(fig, 3.02, ['C1Q⁺ mature', 'ACOD1⁺ inflammatory-like',
                               'NRG1⁺ activated-like'], SUBSET_COLORS)
    draw_bars(fig, counts, 3.21, 4.39, SUBSETS, POOL, SUBSET_COLORS, 'Share of pooled cells')
    aligned_row(fig, LEFT, PLOT_WIDTH, from_top(4.73), counts[POOL].astype(str),
                'Three subsets (n)', size=8.1)
    aligned_row(fig, LEFT, PLOT_WIDTH, from_top(4.93),
                [f'{v:.2f}%' for v in 100 * counts[POOL] / counts[ALL]],
                'Three subsets / all QC cells', size=8.1)

    panel_title(fig, 5.32, 'C', r'$\mathit{TYROBP}$ expression by donor and cell population')
    fig.text(.079, from_top(5.59), 'Donor pseudobulk expression (TMM-normalized log₂ counts per million)',
             color=MUTED, fontsize=8.9, va='center')
    group_headings(fig, 5.88)
    for index, (donor, x) in enumerate(zip(DONORS, X)):
        fig.text(xfig(x, LEFT, PLOT_WIDTH), from_top(6.13), donor, fontsize=8.6,
                 color=donor_color(index), ha='center', va='center')
    draw_expression(fig, counts, expr)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    setup_font()
    counts, expr = read_results()
    fig = combined_figure(counts, expr)
    for suffix in ['pdf', 'png']:
        fig.savefig(args.out / f'{STEM}.{suffix}', dpi=600, facecolor='white',
                    metadata={'Title': 'Donor-level cell composition and TYROBP expression in GSE214695',
                              'Creator': 'GSE214695 donor-level analysis'})
    plt.close(fig)


if __name__ == '__main__':
    main()
'''
}


def _root_module(key, name=None):
    """Load saved source without executing its command-line main block."""
    name = name or ('_tyrobp_' + key)
    mod = types.ModuleType(name)
    mod.__file__ = str(Path(__file__).resolve())
    argv = sys.argv
    try:
        sys.argv = [str(Path(__file__).resolve())]
        exec(compile(_ROOT_SOURCES[key], '<original:'+key+'>', 'exec'), mod.__dict__)
    finally:
        sys.argv = argv
    return mod

def _root_save(fig, out, stem, dpi=600, preview_only=False):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    paths=[]
    formats = [('png',140)] if preview_only else [('pdf',None),('png',dpi)]
    for ext,resolution in formats:
        path=out/(stem+'.'+ext)
        fig.savefig(path, dpi=resolution, facecolor='white')
        paths.append(str(path))
    import matplotlib.pyplot as plt
    plt.close(fig)
    return paths

def run_total_violin(data_root, output_root, dpi=600, preview_only=False, **kwargs):
    import numpy as np
    import pandas as pd
    m=_root_module('total_violin'); m.theme()
    base=Path(data_root)/'macrophages/data'
    df=pd.read_csv(base/'TYROBP_donor_expression.csv')
    stats=pd.read_csv(base/'TYROBP_scope_results.csv')
    d=df[df.scope.eq(m.TOTAL)]
    assert len(d)==12 and d.groupby('group').size().to_dict()=={'Control':6,'UC':6}
    stat=stats[stats.scope.eq(m.TOTAL)].iloc[0]
    assert round(float(stat.PValue),3)==.475
    fig,ax=m.plt.subplots(figsize=(3.7,3.75))
    m.violin(ax,df,m.TOTAL,stat);fig.tight_layout(pad=.65)
    out=Path(output_root)/'fig01_K_total_macrophages'
    outputs=_root_save(fig,out,'Total_macrophages_TYROBP',dpi,preview_only)
    pd.DataFrame(m.PLOT_AUDIT).to_csv(out/'plotted_donor_points.csv',index=False)
    return dict(stage='fig01_K_total_macrophages',outputs=outputs,n_donors=12,P=float(stat.PValue),
                inference='Frozen original donor-level edgeR quasi-likelihood test; not cell-level testing')

def run_donor_figure(data_root, output_root, dpi=600, preview_only=False, **kwargs):
    # Keep the three original modules' native functions and styles intact.
    names=['plot_separate_donor_figures','plot_expression_alternatives']
    previous={n:sys.modules.get(n) for n in names}
    try:
        separate=_root_module('donor_separate',names[0])
        separate.BASE=Path(data_root)/'donors'
        sys.modules[names[0]]=separate
        alternatives=_root_module('donor_alternatives',names[1])
        sys.modules[names[1]]=alternatives
        combined=_root_module('donor_combined')
        combined.setup_font()
        counts,expr=combined.read_results()
        fig=combined.combined_figure(counts,expr)
        out=Path(output_root)/'fig05_donor_composition_expression'
        outputs=_root_save(fig,out,combined.STEM,dpi,preview_only)
        return dict(stage='fig05_donor_composition_expression',outputs=outputs,
                    donors=12,QC_cells=int(counts[combined.ALL].sum()),
                    macrophage_related_cells=int(counts[combined.POOL].sum()),
                    inference='Saved TMM-normalized donor pseudobulk; missing subsets remain missing')
    finally:
        for name,old in previous.items():
            if old is None: sys.modules.pop(name,None)
            else: sys.modules[name]=old

def run_subset_figure(data_root, output_root, dpi=600, preview_only=False,
                      subset_palette='orange_red', **kwargs):
    import numpy as np
    import pandas as pd
    m=_root_module('subset_umap');m.theme()
    # The saved final layout uses a purple scale. The uploaded version uses the
    # earlier native orange-red option; retain both choices explicitly.
    if subset_palette=='orange_red':
        from matplotlib.colors import LinearSegmentedColormap
        m.CMAP=LinearSegmentedColormap.from_list('TYROBP_orange_red',
            [(0,'#DFE3E6'),(.2,'#FDBB6C'),(.4,'#EF7D23'),(.7,'#C74913'),(1,'#7F2704')])
    elif subset_palette!='original_purple':
        raise ValueError('subset_palette must be orange_red or original_purple')
    base=Path(data_root)/'macrophages/data'
    cells=pd.read_csv(base/'TYROBP_cell_expression_and_UMAP.csv')
    expr=pd.read_csv(base/'TYROBP_donor_expression.csv')
    stats=pd.read_csv(base/'TYROBP_scope_results.csv')
    assert len(cells)==1338 and cells.cell.is_unique
    assert np.allclose(cells.TYROBP_log_normalized,np.log1p(1e4*cells.TYROBP_UMI/cells.total_UMI),atol=1e-12)
    limits=[]
    for col in ['UMAP_1','UMAP_2']:
        v=cells[col];pad=np.ptp(v)*.055;limits.append((v.min()-pad,v.max()+pad))
    fig=m.plt.figure(figsize=(11.6,11.15))
    gs=fig.add_gridspec(3,3,width_ratios=[1,1,1.10],wspace=.45,hspace=.65)
    fig.subplots_adjust(left=.075,right=.985,bottom=.14,top=.89)
    row_axes=[]
    for row,scope in enumerate(m.SCOPES):
        axes=[fig.add_subplot(gs[row,col]) for col in range(3)]
        for ax,g in zip(axes[:2],m.GROUPS):m.feature(ax,cells,scope,g,limits)
        stat=stats[stats.scope.eq(scope)].iloc[0]
        m.expression_panel(axes[2],expr,scope,stat);row_axes.append(axes)
    fig.canvas.draw()
    for row,(scope,axes) in enumerate(zip(m.SCOPES,row_axes)):
        y=max(ax.get_position().y1 for ax in axes)+.037
        fig.text(.045,y,chr(65+row),fontsize=14,fontweight='bold',ha='left')
        fig.text(.079,y,m.POP[scope],fontsize=11.5,ha='left')
    fig.suptitle('TYROBP expression in macrophage subsets',x=.51,y=.985,fontsize=15)
    m.bar(fig,[.16,.065,.40,.012])
    fig.text(.69,.062,'Grey points: other myeloid cells',fontsize=9,color='#58636C',ha='left')
    out=Path(output_root)/'fig06_macrophage_subsets'
    outputs=_root_save(fig,out,'Supplementary_TYROBP_subsets_UMAP_'+subset_palette,dpi,preview_only)
    plotted=pd.DataFrame(m.PLOTTED)
    assert len(plotted)==596 and plotted.cell.is_unique
    plotted.to_csv(out/'plotted_feature_points.csv',index=False)
    counts=cells[cells.population.isin(m.POP.values())].groupby(['population','group']).size().unstack(fill_value=0)
    counts.to_csv(out/'observed_cell_counts.csv')
    assert counts.loc[m.POP['C1Q_mature']].to_dict()=={'Control':163,'UC':101}
    assert counts.loc[m.POP['ACOD1_inflammatory_like']].to_dict()=={'Control':0,'UC':160}
    assert counts.loc[m.POP['NRG1_activated_like']].to_dict()=={'Control':4,'UC':168}
    return dict(stage='fig06_macrophage_subsets',outputs=outputs,subset_palette=subset_palette,
                target_cells=596,background_myeloid_cells=1338,
                C1Q_P=float(stats[stats.scope.eq('C1Q_mature')].iloc[0].PValue),
                sparse_subsets='Descriptive only; no missing donor values imputed')

ROOT_PY_STAGES={
    'fig01_K_total_macrophages':run_total_violin,
    'fig05_donor_composition_expression':run_donor_figure,
    'fig06_macrophage_subsets':run_subset_figure,
}

# Unified offline entry point. No stage runs on import.
PY_STAGES.update(ROOT_PY_STAGES)

def run_python_stage(stage, data_root=None, output_root=None, **kwargs):
    base=Path(__file__).resolve().parent
    data_root=Path(data_root) if data_root else base/'data'
    output_root=Path(output_root) if output_root else base/'figures'
    if stage not in PY_STAGES:raise ValueError('Unknown stage: '+stage)
    if stage=='fig03_spatial':
        data_root=data_root/'spatial'
        kwargs.pop('subset_palette',None)
    return PY_STAGES[stage](data_root=data_root,output_root=output_root,**kwargs)

def check_package_inputs(data_root):
    base=Path(__file__).resolve().parent
    failures=[];checked=0
    manifest=json.loads((base/'provenance/input_sha256.json').read_text(encoding='utf-8'))
    for rel,digest in manifest.items():
        path=Path(data_root)/rel
        if not path.is_file():failures.append({'path':rel,'error':'missing'})
        elif _spatial_sha256(path)!=digest:failures.append({'path':rel,'error':'SHA256 changed'})
        checked+=1
    deps={name:importlib.util.find_spec(name) is not None for name in ['numpy','pandas','scipy','matplotlib','PIL']}
    return {'checked_real_input_files':checked,'input_failures':failures,'Python_dependencies':deps,
            'passed':not failures and all(deps.values())}

def main():
    base=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description='Consolidated TYROBP single-cell/spatial native plotting code')
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--stage',choices=['all',*PY_STAGES])
    parser.add_argument('--data-root',type=Path,default=base/'data')
    parser.add_argument('--output-root',type=Path,default=base/'figures')
    parser.add_argument('--dpi',type=int,default=None)
    parser.add_argument('--preview-only',action='store_true')
    parser.add_argument('--subset-palette',choices=['orange_red','original_purple'],default='orange_red')
    parser.add_argument('--font-dir',type=Path,default=None)
    args=parser.parse_args()
    if args.list or (not args.stage and not args.check):
        print(json.dumps({'stages':list(PY_STAGES)},indent=2));return
    if args.check:
        report=check_package_inputs(args.data_root)
        print(json.dumps(report,indent=2));return
    stages=list(PY_STAGES) if args.stage=='all' else [args.stage]
    reports=[]
    for stage in stages:
        dpi=args.dpi if args.dpi is not None else (900 if stage=='fig03_spatial' else 600)
        reports.append(run_python_stage(stage,args.data_root,args.output_root,dpi=dpi,
                       preview_only=args.preview_only,subset_palette=args.subset_palette,font_dir=args.font_dir))
    args.output_root.mkdir(parents=True,exist_ok=True)
    (args.output_root/'python_run_report.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    print(json.dumps(reports,indent=2))

if __name__=='__main__':main()
