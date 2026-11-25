#!/usr/bin/env python3
"""Recovered Python workflow for attached images 4, 6/8 and 9.

The original plotting functions and statistical procedures are retained. Image 6
and image 8 are the same figure. Inputs are observed source tables, never values
digitized from an image. Default figure-4 palette restores the supplied blue/orange
reference; --palette source keeps the later recovered source-file palette.

The GEO-level original analysis scripts are embedded at the end and can be
exported with --export-original-sources for inspection or raw-data reruns. The
default commands operate offline on the included small analysis-input tables.

Examples:
  python transcriptome_python.py --list
  python transcriptome_python.py --check --data-root data/python
  python transcriptome_python.py --stage all --data-root data/python --output-root results/python
  python transcriptome_python.py --stage fig04_gse38713 --recompute
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import importlib.util
import json
import shutil
import warnings

def _fig04_analysis_namespace():
    import argparse
    from hashlib import sha256
    import json
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from scipy.special import expit, logit
    from scipy.stats import mannwhitneyu
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests
    from statsmodels.stats.proportion import proportion_confint

    GENES = ('CASP1', 'MMP9', 'PECAM1', 'TYROBP')
    SEED = 20261111
    FOLDS = 5
    BOOTSTRAPS = 5000

    def fit_model(x, y):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1, solver='lbfgs', max_iter=5000, tol=1e-10))
        model.fit(x, y)
        scaler, regression = (model.steps[0][1], model.steps[1][1])
        if regression.n_iter_[0] >= regression.max_iter:
            raise RuntimeError('Ridge logistic regression failed to converge')
        weights = regression.coef_[0] / scaler.scale_
        intercept = float(regression.intercept_[0] - weights @ scaler.mean_)
        return (model, intercept, weights)

    def cross_validate(x, y, seed):
        predictions = np.full(len(y), np.nan)
        assignments = np.zeros(len(y), dtype=int)
        fits = []
        splits = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(splits.split(x, y), start=1):
            model, intercept, weights = fit_model(x[train], y[train])
            assert not np.any(assignments[test]) and (not np.intersect1d(train, test).size)
            predictions[test] = model.predict_proba(x[test])[:, 1]
            assignments[test] = fold
            assert np.allclose(predictions[test], expit(intercept + x[test] @ weights), rtol=0, atol=1e-12)
            fits.append({'fold': fold, 'n_train': len(train), 'n_test': len(test), 'UC_test': int(y[test].sum()), 'control_test': int((y[test] == 0).sum()), 'intercept_raw_log2': intercept, **dict(zip(GENES, weights))})
        assert np.isfinite(predictions).all() and (assignments > 0).all()
        return (predictions, assignments, fits)

    def auc_interval(y, predictions):
        rng = np.random.default_rng(SEED)
        cases, controls = (predictions[y == 1], predictions[y == 0])
        draws = []
        for _ in range(BOOTSTRAPS):
            pos = rng.choice(cases, len(cases), replace=True)[:, None]
            neg = rng.choice(controls, len(controls), replace=True)[None, :]
            draws.append(float(np.mean((pos > neg) + 0.5 * (pos == neg))))
        lower, upper = np.quantile(draws, [0.025, 0.975])
        return (float(lower), float(upper))

    def calibration_outputs(y, oof, output):
        logits = logit(oof)
        calibration = sm.GLM(y, sm.add_constant(logits), family=sm.families.Binomial()).fit()
        overall = sm.GLM(y, np.ones((len(y), 1)), offset=logits, family=sm.families.Binomial()).fit()
        if not calibration.converged or not overall.converged:
            raise RuntimeError('Calibration regression failed to converge')
        grid = np.linspace(max(0.001, oof.min()), min(0.999, oof.max()), 300)
        pd.DataFrame({'predicted_probability': grid, 'observed_probability': expit(calibration.params[0] + calibration.params[1] * logit(grid))}).to_csv(output / 'calibration_curve.csv', index=False)
        bins = []
        for key in np.unique((bin_ids := pd.qcut(oof, 3, labels=False, duplicates='drop'))):
            mask = bin_ids == key
            positives, count = (int(y[mask].sum()), int(mask.sum()))
            low, high = proportion_confint(positives, count, method='wilson')
            bins.append({'n': count, 'n_uc': positives, 'mean_predicted': oof[mask].mean(), 'observed_fraction': positives / count, 'ci95_lower': low, 'ci95_upper': high})
        pd.DataFrame(bins).to_csv(output / 'calibration_bins.csv', index=False)
        intervals = calibration.conf_int()
        return {'intercept_joint': float(calibration.params[0]), 'slope': float(calibration.params[1]), 'intercept_joint_ci95': intervals[0].tolist(), 'slope_ci95': intervals[1].tolist(), 'calibration_in_the_large': float(overall.params[0]), 'calibration_in_the_large_ci95': overall.conf_int()[0].tolist()}

    def decision_curve(y, oof, output):
        rows = []
        for threshold in np.arange(1, 96) / 100:
            odds = threshold / (1 - threshold)
            positive = oof >= threshold
            tp = np.sum(positive & (y == 1))
            fp = np.sum(positive & (y == 0))
            rows.append({'threshold': threshold, 'model_net_benefit': (tp - odds * fp) / len(y), 'treat_all_net_benefit': (y.sum() - odds * (y == 0).sum()) / len(y), 'treat_none_net_benefit': 0.0})
        pd.DataFrame(rows).to_csv(output / 'decision_curve_oof.csv', index=False)

    def expression_statistics(data, output):
        rows = []
        for gene in GENES:
            control = data.loc[data.group.eq('Control'), gene]
            uc = data.loc[data.group.eq('UC'), gene]
            result = mannwhitneyu(uc, control, alternative='two-sided', method='auto')
            rows.append({'gene': gene, 'control_n': len(control), 'uc_n': len(uc), 'control_median_log2': control.median(), 'uc_median_log2': uc.median(), 'U_statistic': result.statistic, 'p_value': result.pvalue})
        df = pd.DataFrame(rows)
        df['q_bh'] = multipletests(df.p_value, method='fdr_bh')[1]
        df.to_csv(output / 'expression_statistics.csv', index=False)

    def run(inputs, output):
        data = pd.read_csv(inputs)
        assert data.sample_id.is_unique and data.donor_id.is_unique
        assert data.group.value_counts().to_dict() == {'UC': 23, 'Control': 13}
        assert data.activity.value_counts().to_dict() == {'Active UC': 15, 'Control': 13, 'Inactive UC': 8}
        x = data[list(GENES)].to_numpy(dtype=float)
        y = data.UC_status.to_numpy(dtype=int)
        assert np.isfinite(x).all() and np.array_equal(y, data.group.eq('UC').astype(int).to_numpy())
        output.mkdir(parents=True, exist_ok=True)
        oof, folds, fold_fits = cross_validate(x, y, SEED)
        full_model, intercept, weights = fit_model(x, y)
        full_predictions = full_model.predict_proba(x)[:, 1]
        formula_error = float(np.max(np.abs(full_predictions - expit(intercept + x @ weights))))
        assert formula_error < 1e-12
        data.assign(fold_id=folds, p_oof=oof, p_full_fit=full_predictions).to_csv(output / 'sample_predictions.csv', index=False, float_format='%.15g')
        pd.DataFrame(fold_fits).to_csv(output / 'fold_models.csv', index=False)
        scaler, logistic = (full_model.steps[0][1], full_model.steps[1][1])
        pd.DataFrame({'term': ['Intercept', *GENES], 'coefficient_raw_log2': [intercept, *weights], 'coefficient_standardized': [logistic.intercept_[0], *logistic.coef_[0]], 'training_mean': ['NA', *scaler.mean_], 'training_sd_ddof0': ['NA', *scaler.scale_]}).to_csv(output / 'full_model_coefficients.csv', index=False)
        lower, upper = auc_interval(y, oof)
        roc_x, roc_y, thresholds = roc_curve(y, oof, drop_intermediate=False)
        pd.DataFrame({'fpr': roc_x, 'tpr': roc_y, 'threshold': thresholds}).to_csv(output / 'roc_oof.csv', index=False)
        calibration = calibration_outputs(y, oof, output)
        decision_curve(y, oof, output)
        expression_statistics(data, output)
        tp = int(((oof >= 0.5) & (y == 1)).sum())
        tn = int(((oof < 0.5) & (y == 0)).sum())
        performance = {'oof': {'auc': float(roc_auc_score(y, oof)), 'ci95_lower': lower, 'ci95_upper': upper, 'brier_score': float(brier_score_loss(y, oof)), 'cutoff': 0.5, 'sensitivity': tp / 23, 'specificity': tn / 13, 'tp': tp, 'tn': tn, 'fp': 13 - tn, 'fn': 23 - tp}, 'full_fit': {'auc': float(roc_auc_score(y, full_predictions)), 'brier_score': float(brier_score_loss(y, full_predictions))}}
        report = {'dataset': 'GSE38713', 'n_uc': 23, 'n_control': 13, 'activity': {'active_uc': 15, 'remission_uc': 8}, 'input_sha256': sha256(inputs.read_bytes()).hexdigest(), 'input_scale': 'GEO-deposited GPL570 GC-RMA log2, mean across unambiguous probes', 'analysis': 'Single-cohort model development and fixed-panel internal cross-validation', 'model': {'genes': list(GENES), 'method': 'StandardScaler and L2 logistic regression (C=1)', 'raw_log2_intercept': intercept, 'raw_log2_coefficients': dict(zip(GENES, weights)), 'full_fit_formula_max_abs_error': formula_error}, 'validation': {'random_seed': SEED, 'n_stratified_folds': FOLDS, 'scaling_and_estimation_within_training_fold': True, 'auc_interval': f'{BOOTSTRAPS} stratified bootstrap resamples of fixed OOF predictions'}, 'performance': performance, 'calibration': calibration}
        (output / 'model_summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'oof': performance['oof'], 'calibration': calibration, 'intercept': intercept, 'coefficients': dict(zip(GENES, weights))}, indent=2))
    return SimpleNamespace(run=run)

def _fig04_plot_namespace(palette="screenshot"):
    import argparse
    import json
    from pathlib import Path
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, MultipleLocator
    import numpy as np
    import pandas as pd
    from scipy.special import expit, logit

    GENES = ('CASP1', 'MMP9', 'PECAM1', 'TYROBP')
    BLUE = '#14297A'
    RED = '#8C0026'
    INK = '#151920'
    MUTED = '#536577'
    IDEAL = '#677B8C'
    GENE_COLORS = {'CASP1': '#267044', 'MMP9': RED, 'PECAM1': BLUE, 'TYROBP': '#643887'}

    def panel_header(ax, letter, title, subtitle, top=False):
        title_y, subtitle_y = (1.12, 1.056) if top else (1.18, 1.1)
        ax.text(-0.15 if top else -0.17, title_y, letter, transform=ax.transAxes, fontfamily='Times New Roman', fontsize=29, va='center')
        ax.text(0, title_y, title, transform=ax.transAxes, fontsize=15, fontweight='bold', va='center')
        ax.text(0, subtitle_y, subtitle, transform=ax.transAxes, fontsize=9.5, color=MUTED, va='center')

    def expression_panel(ax, data, stats):
        stats = stats.set_index('gene')
        offsets = {'Control': -0.17, 'UC': 0.17}
        rng = np.random.default_rng(20261111)
        bottom = float(data[list(GENES)].min().min())
        top = float(data[list(GENES)].max().max())
        span = top - bottom
        for i, gene in enumerate(GENES):
            for group, color in [('Control', BLUE), ('UC', RED)]:
                values = data.loc[data['group'].eq(group), gene].to_numpy()
                pos = i + offsets[group]
                ax.boxplot(values, positions=[pos], widths=0.25, patch_artist=True, manage_ticks=False, showfliers=False, boxprops={'facecolor': color, 'edgecolor': color, 'lw': 1.2}, whiskerprops={'color': color, 'lw': 1.2}, capprops={'color': color, 'lw': 1.2}, medianprops={'color': 'white', 'lw': 1.7})
                ax.scatter(pos + rng.uniform(-0.11, 0.11, len(values)), values, s=19, facecolor=color, edgecolor='white', lw=0.45, alpha=0.91, zorder=3)
            q = float(stats.loc[gene, 'q_bh'])
            sig = '***' if q < 0.001 else '**' if q < 0.01 else '*' if q < 0.05 else 'ns'
            y = top + 0.038 * span
            lo, hi = (i + offsets['Control'], i + offsets['UC'])
            ax.plot([lo, lo, hi, hi], [y, y + 0.018 * span, y + 0.018 * span, y], color=INK, lw=1)
            ax.text(i, y + 0.028 * span, sig, ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set(xlim=(-0.54, 3.54), ylim=(bottom - 0.055 * span, top + 0.155 * span), ylabel='GC-RMA log2 expression', xticks=range(4), xticklabels=GENES)
        ax.tick_params(axis='x', labelsize=10)
        ax.yaxis.set_major_locator(MultipleLocator(2))
        for tick in ax.get_xticklabels():
            tick.set_fontweight('bold')
        handles = [Line2D([], [], marker='s', ms=7, markerfacecolor=color, markeredgecolor=color, lw=0, label=group) for group, color in [('Control', BLUE), ('UC', RED)]]
        ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1, 1.145), ncols=2, frameon=False, columnspacing=1.0, handletextpad=0.2, borderaxespad=0, fontsize=9.5)
        ax.text(0, -0.15, 'BH-adjusted P: * < 0.05, ** < 0.01, *** < 0.001', transform=ax.transAxes, color=MUTED, fontsize=8.5, va='top')

    def display_number(value):
        return f'{value:.3g}'

    def gene_ticks(lo, hi, points, extent):
        if extent < 12:
            return np.array([])
        if extent < 30:
            mid = round(lo + (hi - lo) / 2, 1)
            return np.array([lo, mid, hi])
        step = next((s for s in (0.5, 1, 2, 2.5, 5, 10, 20) if (hi - lo) / s <= max(1, extent / 8)))
        interior = np.arange(np.ceil(lo / step) * step, hi + 1e-10, step)
        min_spacing = 5.2
        keep = [v for v in interior if min(abs(points(v) - points(lo)), abs(points(v) - points(hi))) >= min_spacing]
        return np.unique(np.r_[lo, keep, hi])

    def nomogram_panel(ax, data, model):
        coefficients = model['raw_log2_coefficients']
        intercept = float(model['raw_log2_intercept'])
        ranges = {g: (np.floor(data[g].min() * 2) / 2, np.ceil(data[g].max() * 2) / 2) for g in GENES}
        minimums = {g: min(coefficients[g] * endpoints[0], coefficients[g] * endpoints[1]) for g, endpoints in ranges.items()}
        spans = {g: abs(coefficients[g]) * (hi - lo) for g, (lo, hi) in ranges.items()}
        scale = 100 / max(spans.values())
        total_max = sum(spans.values()) * scale
        min_lp = intercept + sum(minimums.values())
        total_points = sum((scale * (coefficients[g] * data[g].to_numpy() - minimums[g]) for g in GENES))
        np.testing.assert_allclose(expit(min_lp + total_points / scale), data['p_full_fit'], atol=1e-10, rtol=0)
        left, right = (0.24, 0.99)
        ax.set(xlim=(0, 1), ylim=(0.07, 7.13))
        ax.set_axis_off()

        def line(label, y, domain, values, point_positions, *, extent=None, color=BLUE, bold=False, italic=False, explanation=None):
            x = left + (right - left) * np.asarray(point_positions) / domain
            line_end = max(x) if extent is None else left + (right - left) * extent / domain
            line_start = min(x) if extent is None else left
            ax.text(0, y, label, ha='left', va='center', fontsize=9.8, fontweight='bold' if bold else 'normal', fontstyle='italic' if italic else 'normal')
            ax.plot([line_start, line_end], [y, y], lw=1.75, color=color, solid_capstyle='butt')
            ax.vlines(x, y - 0.115, y, lw=1, color=color)
            for position, value in zip(x, values):
                ax.text(position, y - 0.185, display_number(value), ha='center', va='top', fontsize=8.3)
            if explanation:
                ax.text(line_end + 0.022, y, explanation, ha='left', va='center', fontsize=8.2, color=MUTED)
        point_ticks = np.arange(0, 101, 10)
        line('Points', 6.68, 100, point_ticks, point_ticks, bold=True)
        for gene, y in zip(('MMP9', 'PECAM1', 'CASP1', 'TYROBP'), (5.76, 4.84, 3.92, 3.0)):
            lo, hi = ranges[gene]

            def points(values):
                return scale * (coefficients[gene] * values - minimums[gene])
            extent = scale * spans[gene]
            values = gene_ticks(lo, hi, points, extent)
            explanation = None
            if not len(values):
                first, last = (lo, hi) if coefficients[gene] >= 0 else (hi, lo)
                explanation = f'{first:g} to {last:g} (left to right)'
            line(gene, y, 100, values, points(values), extent=extent, color=GENE_COLORS[gene], italic=True, explanation=explanation)
        ticks = np.arange(0, total_max + 1e-10, 25)
        if total_max - ticks[-1] >= 0.053 * total_max:
            ticks = np.r_[ticks, total_max]
        line('Total points', 1.91, total_max, ticks, ticks, extent=total_max, bold=True)
        probs = np.array([0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999])
        probs_points = scale * (logit(probs) - min_lp)
        chosen = []
        for i in (6, 3, 9, 1, 11, 5, 7, 0, 12, 2, 10, 4, 8):
            point = probs_points[i]
            if 0 <= point <= total_max and all((abs(point - probs_points[j]) >= 0.063 * total_max for j in chosen)):
                chosen.append(i)
        chosen.sort()
        if len(chosen) < 2:
            probs_points = np.linspace(0, total_max, 3)
            probs = expit(min_lp + probs_points / scale)
            chosen = [0, 1, 2]
        line('Predicted UC\nprobability', 0.76, total_max, probs[chosen], probs_points[chosen], color=RED, bold=True)

    def roc_panel(ax, roc, metrics):
        ax.plot([0, 1], [0, 1], color=IDEAL, ls=(0, (4, 3)), lw=1.2)
        ax.plot(roc['fpr'], roc['tpr'], color=BLUE, lw=2.4, drawstyle='steps-post')
        ax.text(0.97, 0.105, f"AUC = {metrics['auc']:.3f}", transform=ax.transAxes, ha='right', va='bottom', fontsize=12)
        ax.text(0.97, 0.034, f"95% CI {metrics['ci95_lower']:.3f}-{metrics['ci95_upper']:.3f}", transform=ax.transAxes, ha='right', va='bottom', fontsize=10)
        ax.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.04), xlabel='1 - specificity', ylabel='Sensitivity')
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))

    def calibration_panel(ax, bins, curve, calibration, metrics):
        ax.plot([0, 1], [0, 1], color=IDEAL, ls=(0, (4, 3)), lw=1.2, label='Ideal calibration')
        if calibration.get('slope') is not None:
            ax.plot(curve['predicted_probability'], curve['observed_probability'], color=BLUE, lw=2.2, label='Logistic calibration')
        errors = np.maximum(0, np.vstack([bins['observed_fraction'] - bins['ci95_lower'], bins['ci95_upper'] - bins['observed_fraction']]))
        ax.errorbar(bins['mean_predicted'], bins['observed_fraction'], yerr=errors, fmt='o', color=RED, lw=1.35, capsize=3, ms=5, label='Risk groups (95% CI)')
        slope_label = f"{calibration['slope']:.3f}" if calibration.get('slope') is not None else 'NA'
        ax.text(0.04, 0.966, f"Slope = {slope_label}\nBrier = {metrics['brier_score']:.3f}", transform=ax.transAxes, ha='left', va='top', fontsize=10.5, linespacing=1.35)
        ax.legend(loc='lower right', frameon=False, fontsize=8.1, handlelength=1.7, borderaxespad=0.45)
        ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.04), xlabel='Predicted probability', ylabel='Observed UC proportion')
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))

    def dca_panel(ax, curve):
        ax.plot(curve['threshold'], curve['model_net_benefit'], lw=2.4, color=BLUE, label='Four-gene model')
        ax.plot(curve['threshold'], curve['treat_all_net_benefit'], lw=1.8, color=RED, label='Treat all')
        ax.plot(curve['threshold'], curve['treat_none_net_benefit'], lw=1.1, color=INK, ls=(0, (4, 3)), label='Treat none')
        maximum = max(curve['model_net_benefit'].max(), curve['treat_all_net_benefit'].max())
        upper = np.ceil((maximum + 0.03) * 10) / 10
        ax.set(xlim=(0, 0.952), ylim=(-0.065, upper), xlabel='Threshold probability', ylabel='Net benefit')
        ax.xaxis.set_major_locator(FixedLocator([0, 0.2, 0.4, 0.6, 0.8, 0.95]))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.legend(loc='lower left', bbox_to_anchor=(0.01, 0.07), frameon=True, edgecolor='none', framealpha=0.91, fontsize=8.4, handlelength=1.7)
    if palette == "screenshot":
        # Restore the existing reference palette, not a different plotting method.
        BLUE, RED = "#0072B2", "#D55E00"
        GENE_COLORS = {"CASP1": RED, "MMP9": RED, "PECAM1": BLUE, "TYROBP": BLUE}
    elif palette != "source":
        raise ValueError("palette must be screenshot or source")
    return SimpleNamespace(panel_header=panel_header, expression_panel=expression_panel,
        nomogram_panel=nomogram_panel, roc_panel=roc_panel,
        calibration_panel=calibration_panel, dca_panel=dca_panel,
        INK=INK, MUTED=MUTED)

def run_fig04_gse38713(input_dir, output_dir, *, recompute=False, palette="screenshot"):
    """Replot the five panels; optionally recompute fixed-panel CV and all statistics."""
    import json
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    input_dir, results = Path(input_dir), Path(output_dir)
    results.mkdir(parents=True, exist_ok=True)
    if recompute:
        _fig04_analysis_namespace().run(input_dir/"model_inputs.csv", results)
    else:
        for filename in PY_STAGES["fig04_gse38713"]["required_inputs"]:
            if filename != "model_inputs.csv":
                shutil.copy2(input_dir/filename, results/filename)
    ns = _fig04_plot_namespace(palette)
    panel_header, expression_panel = ns.panel_header, ns.expression_panel
    nomogram_panel, roc_panel = ns.nomogram_panel, ns.roc_panel
    calibration_panel, dca_panel = ns.calibration_panel, ns.dca_panel
    INK, MUTED = ns.INK, ns.MUTED
    figure_dir = results / 'figures'
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((results / 'model_summary.json').read_text(encoding='utf-8'))
    data = pd.read_csv(results / 'sample_predictions.csv')
    counts = data['group'].value_counts()
    assert counts.to_dict() == {'UC': 23, 'Control': 13}
    cohort = f"GSE38713  |  UC n={counts['UC']}, control n={counts['Control']}"
    metrics = summary['performance']['oof']
    mpl_style = {'font.family': 'Arial', 'font.size': 10, 'pdf.fonttype': 42, 'axes.linewidth': 1.1, 'axes.spines.top': False, 'axes.spines.right': False, 'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'axes.labelsize': 11, 'xtick.labelsize': 9.8, 'ytick.labelsize': 9.8, 'xtick.major.size': 4, 'ytick.major.size': 4, 'savefig.facecolor': 'white'}
    with plt.rc_context(mpl_style):
        figure = plt.figure(figsize=(13.6, 9.4))
        root = figure.add_gridspec(nrows=2, ncols=1, height_ratios=[1.1, 1], left=0.077, right=0.965, top=0.876, bottom=0.108, hspace=0.53)
        top = root[0].subgridspec(1, 2, width_ratios=[0.96, 1.04], wspace=0.28)
        bottom = root[1].subgridspec(1, 3, wspace=0.38)
        a, b = [figure.add_subplot(p) for p in top]
        c, d, e = [figure.add_subplot(p) for p in bottom]
        panel_header(a, 'A', 'Gene expression', cohort, top=True)
        expression_panel(a, data, pd.read_csv(results / 'expression_statistics.csv'))
        panel_header(b, 'B', 'Four-gene nomogram', cohort + '  |  Full-cohort refit', top=True)
        nomogram_panel(b, data, summary['model'])
        panel_header(c, 'C', 'ROC curve', cohort)
        roc_panel(c, pd.read_csv(results / 'roc_oof.csv'), metrics)
        panel_header(d, 'D', 'Calibration', cohort)
        calibration_panel(d, pd.read_csv(results / 'calibration_bins.csv'), pd.read_csv(results / 'calibration_curve.csv'), summary['calibration'], metrics)
        panel_header(e, 'E', 'Decision curve', cohort)
        dca_panel(e, pd.read_csv(results / 'decision_curve_oof.csv'))
        notes = ['Five-fold out-of-fold predictions', 'Five-fold out-of-fold predictions', 'Five-fold out-of-fold predictions (exploratory)']
        for axis, note in zip([c, d, e], notes):
            axis.text(0, -0.23, note, transform=axis.transAxes, va='top', fontsize=8.6, color=MUTED)
        stem = figure_dir / 'GSE38713_four_gene_refit'
        figure.savefig(stem.with_suffix('.pdf'))
        figure.savefig(stem.with_suffix('.png'), dpi=600)
        plt.close(figure)

def _fig06_violin_namespace():
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MaxNLocator
    from scipy.stats import mannwhitneyu, false_discovery_control, gaussian_kde

    GENES = ('TYROBP', 'CASP1', 'MMP9', 'PECAM1')
    INK = '#1F2A33'
    BLUE = '#153D61'
    RED = '#A4213F'

    def draw_group(ax, center, values, color, bandwidth, seed, mark_scale=1):
        y = np.linspace(values.min(), values.max(), 400)
        kde = gaussian_kde(values, bw_method=bandwidth / values.std(ddof=1))
        density = kde(y)
        widths = 0.37 * density / density.max()
        ax.fill_betweenx(y, center - widths, center + widths, facecolor=color, edgecolor=color, linewidth=0.65, zorder=1)
        q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        iqr = q3 - q1
        whisk_lo = values[values >= q1 - 1.5 * iqr].min()
        whisk_hi = values[values <= q3 + 1.5 * iqr].max()
        ax.vlines(center, whisk_lo, whisk_hi, colors=INK, linewidth=0.75, zorder=2)
        w = 0.115
        ax.add_patch(Rectangle((center - w / 2, q1), w, iqr, facecolor='white', edgecolor=INK, linewidth=0.9, zorder=3))
        ax.hlines(med, center - w / 2, center + w / 2, color=INK, linewidth=1.15, zorder=4)
        rng = np.random.default_rng(seed)
        placed = []
        trans = ax.transData
        for i in np.argsort(values, kind='stable'):
            value = values[i]
            max_x = min(0.205, float(np.interp(value, y, widths)) * 0.91)
            start = rng.uniform(-max_x, max_x)
            options = np.r_[start, np.linspace(-max_x, max_x, 73)]
            xy = trans.transform(np.c_[center + options, np.full(options.shape, value)])
            if placed:
                close = np.sqrt(((xy[:, None, :] - np.array(placed)[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
            else:
                close = np.full(len(options), 100000.0)
            spacing = 5.9 * mark_scale * ax.figure.dpi / 72
            pref = np.abs(options - start) + (0.065 * (np.abs(options) < 0.065) if q1 <= value <= q3 else 0)
            viable = np.flatnonzero(close >= spacing)
            choice = viable[np.argmin(pref[viable])] if len(viable) else np.argmax(close - pref * 0.01)
            placed.append(xy[choice])
            ax.scatter(center + options[choice], value, s=25 * mark_scale ** 2, facecolors=color, edgecolors='white', linewidths=0.78 * mark_scale, zorder=5)

    def format_panel(ax, gene, comp, pair, adjusted, p_symbol):
        lo = min((a.min() for a in pair))
        hi = max((a.max() for a in pair))
        span = hi - lo
        lower, upper = (lo - 0.145 * span, hi + 0.36 * span)
        ax.set(xlim=(-0.55, 1.55), ylim=(lower, upper))
        ticks = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(lower, hi + 0.1 * span)
        ticks = ticks[(ticks >= lower) & (ticks <= hi + 0.12 * span)]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{x:.1f}' if len(ticks) > 1 and ticks[1] - ticks[0] < 1 else f'{x:g}' for x in ticks])
        labels = ('CD', 'UC') if comp == 'CDc' else ('Control', 'UC')
        ax.set_xticks([0, 1], [f'{name}\n(n = {len(a)})' for name, a in zip(labels, pair)])
        ax.tick_params(axis='x', length=0, pad=4, labelsize=10.7)
        ax.tick_params(axis='y', length=3.5, width=0.9, pad=3.5, labelsize=10)
        ax.set_ylabel('Gene expression (log$_2$ scale)', labelpad=6, fontsize=10.7)
        ax.set_title(gene, fontsize=14.7, fontstyle='italic', pad=9)
        bracket = hi + 0.175 * span
        ax.plot([0, 0, 1, 1], [bracket - 0.04 * span, bracket, bracket, bracket - 0.04 * span], lw=0.85, color=INK, solid_capstyle='butt')
        label = p_symbol + (f' = {adjusted:.3f}' if adjusted >= 0.001 else ' < 0.0001')
        ax.text(0.5, bracket + 0.032 * span, label, ha='center', va='bottom', fontsize=11.1)
    return SimpleNamespace(draw_group=draw_group, format_panel=format_panel, GENES=GENES, INK=INK, BLUE=BLUE, RED=RED)

def _fig06_check_namespace():
    from pathlib import Path
    import matplotlib
    from matplotlib import pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter
    import numpy as np
    import pandas as pd
    from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedKFold

    def check_results(cohort, result):
        name = result.name
        y = cohort['UC_positive'].to_numpy(dtype=int)
        score = cohort[name + '_score'].to_numpy()
        probability = cohort[name + '_probability'].to_numpy()
        np.testing.assert_allclose(1 / (1 + np.exp(-score)), probability)
        predicted = probability >= result.threshold
        counts = confusion_matrix(y, predicted, labels=[1, 0])
        tp, fn, fp, tn = counts.ravel()
        observed_auc = roc_auc_score(y, score)
        np.testing.assert_array_equal([tp, fn, fp, tn], result[['TP', 'FN', 'FP', 'TN']])
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        np.testing.assert_allclose([sens, spec, 0.5 * (sens + spec), observed_auc], result[['sensitivity', 'specificity', 'balanced_accuracy', 'auc']].to_numpy(float))
        pos, neg = (score[y == 1], score[y == 0])
        rng = np.random.default_rng(20261118)
        pp = rng.choice(pos, (5000, len(pos)))
        nn = rng.choice(neg, (5000, len(neg)))
        cmp = pp[:, :, None] - nn[:, None, :]
        auc_ci = np.quantile(((cmp > 0) + 0.5 * (cmp == 0)).mean(axis=(1, 2)), [0.025, 0.975])
        np.testing.assert_allclose(auc_ci, [result.ci_low, result.ci_high], atol=1e-12)
        fpr, tpr, _ = roc_curve(y, score, drop_intermediate=False)
        assert np.any(np.isclose(fpr, 1 - spec) & np.isclose(tpr, sens))
        return (counts, fpr, tpr)
    return check_results

def run_fig06_08_disease_control(input_dir, output_dir, **kwargs):
    """Original combined AB figure; BH P values and OOF AUCs are audited from tables."""
    from pathlib import Path
    import re
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter
    import numpy as np
    import pandas as pd
    from scipy.stats import false_discovery_control, mannwhitneyu
    DATA = Path(input_dir)
    ns = _fig06_violin_namespace()
    draw_group, format_panel, GENES = ns.draw_group, ns.format_panel, ns.GENES
    BLUE, RED, INK = ns.BLUE, ns.RED, ns.INK
    check_results = _fig06_check_namespace()
    # Restore the rcParams side effects of the two original imported plot
    # modules before applying the combined AB source's Times New Roman style.
    plt.rcParams.update({
        "font.family":"Arial", "font.size":10, "text.color":"#26333A",
        "axes.labelcolor":"#26333A", "axes.edgecolor":"#899497", "axes.linewidth":.65,
        "xtick.color":"#535F64", "ytick.color":"#535F64",
        "figure.facecolor":"white", "axes.facecolor":"white", "savefig.facecolor":"white",
        "axes.spines.top":False, "axes.spines.right":False,
        "pdf.fonttype":42, "ps.fonttype":42,
    })
    FONT = "Times New Roman"
    font_loaded = False
    for file in ("times.ttf", "timesi.ttf"):
        font_path = Path("C:/Windows/Fonts")/file
        if font_path.is_file():
            font_manager.fontManager.addfont(str(font_path))
            font_loaded = True
    if not font_loaded:
        available_fonts = {f.name for f in font_manager.fontManager.ttflist}
        if FONT not in available_fonts:
            FONT = "Liberation Serif" if "Liberation Serif" in available_fonts else "DejaVu Serif"
            warnings.warn("Times New Roman unavailable; using an installed serif fallback")
    plt.rcParams.update({
        "font.family": FONT, "font.weight": "normal", "font.size": 9,
        "axes.titleweight": "normal", "axes.labelweight": "normal",
        "axes.linewidth": .75, "text.color": INK, "axes.edgecolor": INK,
        "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": FONT+":italic",
        "mathtext.bf": FONT, "mathtext.fallback": None, "mathtext.cal": FONT,
        "mathtext.sf": FONT, "mathtext.tt": FONT, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    samples=pd.read_csv(DATA/"selected_samples_expression_predictions.csv",float_precision="round_trip")
    cohort=samples[samples.group.isin(["CDc","UC"])]
    assert len(cohort)==cohort.patient_id.nunique()==43 and cohort.UC_positive.sum()==24
    stats=pd.read_csv(DATA/"model_performance.csv").set_index("model")
    saved=pd.read_csv(DATA/"four_gene_expression_UC_vs_CDc.csv").set_index("gene")

    w,h=8.05,4.36
    fig=plt.figure(figsize=(w,h))


    def words(x,y,string,**kwargs):
        return fig.text(x/w,y/h,string,**kwargs)


    def axes(x,y,width,height):
        return fig.add_axes((x/w,y/h,width/w,height/h))


    words(.075,4.24,"A",fontsize=15,va="top")
    words(.36,4.22,"Gene expression in CD and UC",fontsize=11.6,va="top")
    words(4.50,4.24,"B",fontsize=15,va="top")
    words(4.80,4.22,"ROC curves for UC versus CD",fontsize=11.6,va="top")

    pairs=[tuple(samples.loc[samples.group==k,g].to_numpy() for k in ("CDc","UC")) for g in GENES]
    raw=[mannwhitneyu(a,b,alternative="two-sided").pvalue for a,b in pairs]
    adj=false_discovery_control(raw)
    np.testing.assert_allclose(adj,saved.loc[list(GENES),"BH_adjusted_p"])
    for i,(g,pair,p) in enumerate(zip(GENES,pairs,adj)):
        ax=axes([.55,2.62][i%2],[2.40,.43][i//2],1.48,1.27)
        format_panel(ax,g,"CDc",pair,p,r"$P$")
        ax.set_title(g,fontsize=10.8,fontstyle="italic",pad=3)
        ax.set_ylabel(r"Gene expression (log$_2$ scale)",fontsize=8.6,labelpad=2.5)
        ax.tick_params(axis="x",labelsize=8.6,pad=2)
        ax.tick_params(axis="y",labelsize=8.25,pad=2,length=2.6,width=.7)
        for label in ax.texts:
            label.set_fontsize(8.5)
        groups=[part[g].to_numpy() for _,part in samples.groupby("group")]
        centered=np.concatenate([v-v.mean() for v in groups])
        bandwidth=np.sqrt((centered**2).sum()/(len(centered)-len(groups)))*np.median([len(v) for v in groups])**(-.2)
        for j,(v,color) in enumerate(zip(pair,[BLUE,RED])):
            draw_group(ax,j,v,color,bandwidth,136+11*i+j,mark_scale=.58)

    words(6.37,3.94,"GSE16879  ·  Five-fold cross-validation  ·  43 patients",
          ha="center",va="center",fontsize=8.5)
    ax=axes(5.09,1.04,2.67,2.67)
    ax.set(xlim=(-.012,1.012),ylim=(-.012,1.012),aspect="equal")
    ax.set_xlabel("1 − specificity",fontsize=9.4,labelpad=3)
    ax.set_ylabel("Sensitivity",fontsize=9.4,labelpad=3)
    for side in ("left","bottom"):
        ax.spines[side].set_bounds(0,1)
    for axis in (ax.xaxis,ax.yaxis):
        axis.set_ticks(np.linspace(0,1,6))
        axis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.tick_params(labelsize=8.5,length=2.6,width=.7,pad=2.5)
    ax.plot([0,1],[0,1],color="#9EA8AC",lw=.8,ls=(0,(3,3)),zorder=1)

    words(7.85,.568,"AUC (95% CI)",ha="right",fontsize=8.5,va="center")
    for key,name,color,ls,y in [
        ("TYROBP_CV","TYROBP single-gene model",BLUE,"solid",.376),
        ("Four_gene_CV","Four-gene model",RED,(0,(4,2)),.167),
    ]:
        est=stats.loc[key]
        _,fpr,tpr=check_results(cohort,est)
        np.testing.assert_allclose(np.trapezoid(tpr,fpr),est.auc)
        ax.plot(fpr,tpr,color=color,lw=1.55,ls=ls,solid_joinstyle="miter",zorder=2)
        fig.add_artist(Line2D([4.71/w,4.98/w],[y/h,y/h],transform=fig.transFigure,
                              color=color,lw=1.55,ls=ls))
        words(5.04,y,name,fontsize=8.75,va="center")
        words(7.85,y,f"{est.auc:.3f} ({est.ci_low:.3f}–{est.ci_high:.3f})",
              fontsize=8.75,ha="right",va="center")

    out=Path(output_dir)
    out.mkdir(exist_ok=True,parents=True)
    base=out/"Figure_AB_expression_ROC"
    fig.savefig(base.with_suffix(".png"),dpi=600)
    fig.savefig(base.with_suffix(".pdf"),metadata={
        "Title":"Gene expression and ROC curves for UC versus colonic Crohn's disease",
        "Subject":"Panel A shows BH-adjusted P values. Panel B shows pooled out-of-fold ROC curves.",
    })
    plt.close(fig)
    names=set(re.findall(rb"/BaseFont\s*/([^\s/<]+)",base.with_suffix(".pdf").read_bytes()))
    if font_loaded:
        assert names and all(b"TimesNewRoman" in n and b"Bold" not in n for n in names),names
    print(sorted(names))


def _fig09_clinical_namespace(input_dir, output_dir):
    from pathlib import Path
    import json
    import hashlib
    import numpy as np
    import pandas as pd
    from scipy.stats import gaussian_kde
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    ROOT, OUT = Path(input_dir), Path(output_dir)
    NAVY, WINE, PALE = "#173F63", "#B01D42", "#8DA9BF"
    INK = '#202C35'
    SEED = 20261115

    def p_label(value, adjusted=True):
        name = 'P'
        if value < 0.001:
            return f'${name} < 0.001$'
        return f'${name} = {value:.2f}$'

    def theme():
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5, 'axes.labelsize': 11, 'axes.titlesize': 10.5, 'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': INK, 'ytick.color': INK, 'axes.edgecolor': INK, 'axes.linewidth': 1.0, 'xtick.major.size': 3.3, 'ytick.major.size': 3.3, 'xtick.major.width': 0.85, 'ytick.major.width': 0.85, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white', 'figure.facecolor': 'white', 'mathtext.fontset': 'dejavusans'})

    def bracket(ax, x1, x2, y, height, text, size=9.5):
        ax.plot([x1, x1, x2, x2], [y - height, y, y, y - height], lw=0.85, c=INK, clip_on=False)
        ax.text((x1 + x2) / 2, y + height * 0.35, text, ha='center', va='bottom', fontsize=size)

    def violin(ax, arrays, names, colors, limits, comparisons, rng, title=None, letter=None):
        arrays = [np.asarray(v, dtype=float) for v in arrays]
        assert all((len(v) > 1 and np.isfinite(v).all() for v in arrays))
        positions = np.arange(len(arrays))
        bodies = ax.violinplot(arrays, positions=positions, widths=0.72, showmeans=False, showmedians=False, showextrema=False, points=256, bw_method=0.45)
        for body, color in zip(bodies['bodies'], colors):
            body.set(facecolor=color, edgecolor=color, alpha=1, linewidth=0.65, zorder=1)
        for i, (v, color) in enumerate(zip(arrays, colors)):
            density = gaussian_kde(v, bw_method=0.45)(v)
            spread = 0.16 * density / density.max()
            jitter = rng.uniform(-1, 1, len(v)) * spread
            if len(v) <= 10:
                jitter = np.where(np.abs(jitter) < 0.082, np.where(jitter >= 0, 0.082, -0.082), jitter)
            x = i + jitter
            ax.scatter(x, v, s=6 if len(v) > 30 else 9, facecolor=color, edgecolor='white', linewidth=0.24, alpha=0.92, zorder=3)
        boxes = ax.boxplot(arrays, positions=positions, widths=0.115, patch_artist=True, whis=1.5, showfliers=False, showcaps=False, boxprops={'facecolor': 'white', 'edgecolor': INK, 'linewidth': 0.85, 'zorder': 5}, medianprops={'color': INK, 'linewidth': 1.7, 'zorder': 6}, whiskerprops={'color': INK, 'linewidth': 0.8, 'zorder': 4})
        ax.set_xticks(positions, [f'{label}\n(n = {len(v)})' for label, v in zip(names, arrays)])
        ax.set_ylabel('TYROBP expression (log$_2$)')
        ax.set_xlim(-0.57, len(arrays) - 0.43)
        ax.set_ylim(*limits)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(False)
        for x1, x2, y, label in comparisons:
            bracket(ax, x1, x2, y, (limits[1] - limits[0]) * 0.019, label)
        if title:
            ax.set_title(title, pad=9, fontweight='normal')
        if letter:
            ax.annotate(letter, xy=(0, 1), xycoords='axes fraction', xytext=(-33, 9), textcoords='offset points', ha='left', va='bottom', fontsize=15, fontweight='bold', annotation_clip=False)

    def save(fig, stem):
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(OUT / f'{stem}.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.06)
        if stem == '00_TYROBP_clinical_violin_panels_A-E':
            fig.savefig(OUT / f'{stem}_preview.png', dpi=150, bbox_inches='tight', pad_inches=0.06)
        plt.close(fig)

    def main():
        OUT.mkdir(parents=True, exist_ok=True)
        theme()
        stats_files = [ROOT / 'tables' / 'activity_comparisons.csv', ROOT / 'tables' / 'baseline_response_comparisons.csv', ROOT / 'tables' / 'GSE75214_analyzed_samples.csv']
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in stats_files}
        activity = pd.read_csv(stats_files[2])
        activity_tests = pd.read_csv(stats_files[0])
        response_tests = pd.read_csv(stats_files[1])
        groups = ['Control', 'Inactive UC', 'Active UC']
        av = [activity.loc[activity.group.eq(g), 'TYROBP_log2'].to_numpy() for g in groups]
        assert [len(x) for x in av] == [11, 23, 74]
        span = max(np.ptp(np.concatenate(av)), 1.5)
        ymax = np.max(np.concatenate(av))
        activity_limits = (np.min(np.concatenate(av)) - 0.1 * span, ymax + 0.52 * span)
        acomp = []
        for order, (i, j, idx) in enumerate([(0, 1, 0), (1, 2, 2), (0, 2, 1)]):
            p = activity_tests.iloc[idx]['p_Holm_3_activity_comparisons']
            acomp.append((i, j, ymax + span * (0.1 + 0.145 * order), p_label(p)))
        rng = np.random.default_rng(SEED)
        fig, ax = plt.subplots(figsize=(4.7, 4.1))
        violin(ax, av, groups, [NAVY, PALE, WINE], activity_limits, acomp, rng)
        fig.tight_layout(pad=0.6)
        save(fig, '01_TYROBP_activity_violin')
        records = []
        data = {}
        for _, row in response_tests.iterrows():
            key = f'{row.dataset}_{row.endpoint}'
            if row.dataset == 'GSE16879':
                d = pd.read_csv(ROOT / 'datasets' / 'gse16879' / 'UC_baseline_TYROBP.csv')
                values = [d.loc[d.response_group.eq(g), 'TYROBP_log2'].to_numpy() for g in ['Responder', 'Nonresponder']]
            else:
                d = pd.read_csv(ROOT / 'datasets' / 'gse73661' / 'results' / f'{row.endpoint}_paired_samples.csv')
                values = [d.loc[d.response.eq(g), 'TYROBP_log2_RMA_baseline'].to_numpy() for g in ['Responder', 'Nonresponder']]
            assert [len(x) for x in values] == [int(row.n_R), int(row.n_NR)]
            data[key] = (row, values)
        all736 = np.concatenate([v for key, (_, values) in data.items() if key.startswith('GSE73661') for v in values])
        for index, (key, (row, values)) in enumerate(data.items()):
            joint = all736 if row.dataset == 'GSE73661' else np.concatenate(values)
            spread = max(np.ptp(joint), 1.5)
            low, high = (np.min(joint) - 0.11 * spread, np.max(joint) + 0.26 * spread)
            primary = row.family == 'primary_baseline'
            p = row.p_Holm_4_primary_baseline_comparisons if primary else row.p_raw
            comp = [(0, 1, np.max(joint) + 0.15 * spread, p_label(p, primary))]
            fig, ax = plt.subplots(figsize=(3.7, 3.6))
            violin(ax, values, ['Responder', 'Nonresponder'], [NAVY, WINE], (low, high), comp, rng)
            fig.tight_layout(pad=0.6)
            stem = f'{index + 3:02d}_{key}_violin'
            save(fig, stem)
            records.append({'file_stem': stem, 'dataset': row.dataset, 'endpoint': row.endpoint, 'n_responder': int(row.n_R), 'n_nonresponder': int(row.n_NR), 'P_display_type': 'Holm-adjusted across four primary baseline comparisons' if primary else 'Raw P; secondary endpoint', 'P_value': float(p), 'P_label': p_label(p, primary), 'short_title': f"{row.dataset} · {('IFX' if row.treatment == 'Infliximab' else 'VDZ')} " + {'IFX_W4_W6': 'W4–6', 'VDZ_W6': 'W6', 'VDZ_W12': 'W12', 'VDZ_W52': 'W52'}[row.endpoint], 'limits': (low, high), 'comparisons': comp, 'key': key})
        primary = [r for r in records if 'four primary' in r['P_display_type']]
        fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.6))
        for ax, rec in zip(axes.ravel(), primary):
            violin(ax, data[rec['key']][1], ['Responder', 'Nonresponder'], [NAVY, WINE], rec['limits'], rec['comparisons'], rng, title=rec['short_title'])
        fig.subplots_adjust(hspace=0.4, wspace=0.38, left=0.1, right=0.985, bottom=0.09, top=0.95)
        save(fig, '02_TYROBP_response_violin')
        fig = plt.figure(figsize=(7.8, 10.7))
        grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.49, wspace=0.39, left=0.1, right=0.98, bottom=0.065, top=0.97)
        a = fig.add_subplot(grid[0, :])
        violin(a, av, groups, [NAVY, PALE, WINE], activity_limits, acomp, rng, title='GSE75214', letter='A')
        for ax, rec, letter in zip([fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1])], primary, 'BCDE'):
            violin(ax, data[rec['key']][1], ['Responder', 'Nonresponder'], [NAVY, WINE], rec['limits'], rec['comparisons'], rng, title=rec['short_title'], letter=letter)
        save(fig, '00_TYROBP_clinical_violin_panels_A-E')
        pd.DataFrame([{k: v for k, v in r.items() if k not in ['limits', 'comparisons', 'key']} for r in records]).to_csv(OUT / 'figure_panel_map.csv', index=False, encoding='utf-8-sig')
        caption = 'Publication-style violin plots / 小提琴图说明\n\nStyle: dark navy and burgundy matching the supplied reference; pale blue marks\ninactive UC in the three-group panel. White boxes show the interquartile range,\nblack lines the median, whiskers extend to 1.5 IQR, and dots are individual samples.\nDensity curves are truncated at the observed expression range. No data or tests\nwere changed. Sample sizes remain visible, including the n=3 responder group.\n\nMinimal main panels A–E:\nA: GSE75214 colonic Control (11), Inactive UC (23), Active UC (74).\nB: GSE16879 infliximab, response at weeks 4–6 (8 responders / 16 nonresponders).\nC: GSE73661 infliximab, response at weeks 4–6 (8 / 15).\nD: GSE73661 vedolizumab, response at week 6 (6 / 21).\nE: GSE73661 vedolizumab, response at week 12 (3 / 10).\n\nP values in main panels A–E are Holm-adjusted: three pairwise activity contrasts\nin panel A and four primary baseline response comparisons in panels B–E.\nThe display label is simply P. Exact values remain in the statistical CSV\ntables. Values below 0.001 are shown as P < 0.001.\nThe separate week-52 vedolizumab figure is secondary and displays raw P=0.13.\nDo not relabel adjusted P as raw P when assembling the manuscript.\n\nExpression scale: RMA log2 intensity for GSE75214 and GSE73661;\nlog2(GCOS globally scaled signal) for GSE16879. Absolute intensity values are\nnot directly comparable between datasets. GSE73661 baseline panels share a\ncommon y-range. R/NR labels refer to source-defined response, not randomization.\nViolin density for n=3 is descriptive only and does not imply a stable population\ndistribution. All individual observations and n values are shown.\n\n中文：图A为疾病活动度三组比较；图B–E为治疗前表达与后续应答的比较。\n主图A–E中的P值均经过Holm校正，图上统一仅标注P。仅保留组名、样本量、表达纵轴和P值；拼图另保留面板字母\n及简短的数据集/药物/时间标签。第52周单图为次要结果，P为未校正值。\n白箱为四分位距，黑线为中位数，散点为各样本，曲线仅用于描述分布。\nIFX=英夫利昔单抗，VDZ=维多珠单抗。使用图时保留此统计说明于图注。\n\nRun from the analysis folder: python beautify_figures.py\n'
        (OUT / 'Figure_captions_ZH_EN.txt').write_text(caption, encoding='utf-8-sig')
        assert all((hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in before.items()))
    return main

def verify_clinical_statistics(input_dir):
    """Verify the saved Holm corrections directly from the observed sample tables."""
    import numpy as np
    import pandas as pd
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests
    root = Path(input_dir)
    activity = pd.read_csv(root/"tables/GSE75214_analyzed_samples.csv")
    saved_a = pd.read_csv(root/"tables/activity_comparisons.csv")
    raw = []
    for lo, hi in [("Control", "Inactive UC"), ("Control", "Active UC"), ("Inactive UC", "Active UC")]:
        a = activity.loc[activity.group.eq(lo), "TYROBP_log2"]
        b = activity.loc[activity.group.eq(hi), "TYROBP_log2"]
        raw.append(mannwhitneyu(b,a,alternative="two-sided",method="auto").pvalue)
    np.testing.assert_allclose(multipletests(raw, method="holm")[1], saved_a.p_Holm_3_activity_comparisons)
    saved_r = pd.read_csv(root/"tables/baseline_response_comparisons.csv")
    raw = []
    primary = saved_r[saved_r.family.eq("primary_baseline")]
    for row in primary.itertuples():
        if row.dataset == "GSE16879":
            d = pd.read_csv(root/"datasets/gse16879/UC_baseline_TYROBP.csv")
            a = d.loc[d.response_group.eq("Responder"), "TYROBP_log2"]
            b = d.loc[d.response_group.eq("Nonresponder"), "TYROBP_log2"]
        else:
            d = pd.read_csv(root/f"datasets/gse73661/results/{row.endpoint}_paired_samples.csv")
            a = d.loc[d.response.eq("Responder"), "TYROBP_log2_RMA_baseline"]
            b = d.loc[d.response.eq("Nonresponder"), "TYROBP_log2_RMA_baseline"]
        assert [len(a),len(b)] == [row.n_R,row.n_NR]
        raw.append(mannwhitneyu(b,a,alternative="two-sided",method="auto").pvalue)
    np.testing.assert_allclose(multipletests(raw,method="holm")[1],primary.p_Holm_4_primary_baseline_comparisons)


def run_fig09_clinical(input_dir, output_dir, **kwargs):
    verify_clinical_statistics(input_dir)
    return _fig09_clinical_namespace(input_dir, output_dir)()


PY_STAGES = {
    'fig04_gse38713': {'panels': 'Attached image 4, A-E', 'required_inputs': ['model_inputs.csv', 'sample_predictions.csv', 'model_summary.json', 'expression_statistics.csv', 'roc_oof.csv', 'calibration_bins.csv', 'calibration_curve.csv', 'decision_curve_oof.csv'], 'dependencies': ['numpy', 'pandas', 'scipy', 'matplotlib', 'sklearn', 'statsmodels'], "runner": run_fig04_gse38713},
    'fig06_08_disease_control': {'panels': 'Attached images 6 and 8, A-B (duplicate image)', 'required_inputs': ['selected_samples_expression_predictions.csv', 'model_performance.csv', 'four_gene_expression_UC_vs_CDc.csv'], 'dependencies': ['numpy', 'pandas', 'scipy', 'matplotlib', 'sklearn'], "runner": run_fig06_08_disease_control},
    'fig09_clinical': {'panels': 'Attached image 9, A-E', 'required_inputs': ['tables/GSE75214_analyzed_samples.csv', 'tables/activity_comparisons.csv', 'tables/baseline_response_comparisons.csv', 'datasets/gse16879/UC_baseline_TYROBP.csv', 'datasets/gse73661/results/IFX_W4_W6_paired_samples.csv', 'datasets/gse73661/results/VDZ_W6_paired_samples.csv', 'datasets/gse73661/results/VDZ_W12_paired_samples.csv', 'datasets/gse73661/results/VDZ_W52_paired_samples.csv'], 'dependencies': ['numpy', 'pandas', 'scipy', 'matplotlib', 'statsmodels'], "runner": run_fig09_clinical},
}


def list_python_stages():
    return [{"stage": key, **{k: v for k, v in value.items() if k != "runner"}} for key,value in PY_STAGES.items()]


def _resolve_python_inputs(data_root):
    root = Path(data_root)
    if (root/"python"/"fig04_gse38713").is_dir():
        return root/"python"
    return root


def check_python_inputs(data_root, stages=None):
    root = _resolve_python_inputs(data_root)
    result = []
    for key in stages or PY_STAGES:
        entry = PY_STAGES[key]
        result.append({"stage":key,
            "missing_inputs":[str(root/key/name) for name in entry["required_inputs"] if not (root/key/name).is_file()],
            "missing_dependencies":[name for name in entry["dependencies"] if importlib.util.find_spec(name) is None]})
    return result


def run_python_stage(stage, data_root, output_root, *, recompute=False, palette="screenshot"):
    if stage not in PY_STAGES:
        raise KeyError(f"Unknown stage: {stage}; choose from {list(PY_STAGES)}")
    root = _resolve_python_inputs(data_root)
    checks = check_python_inputs(root, [stage])[0]
    if checks["missing_inputs"] or checks["missing_dependencies"]:
        raise RuntimeError(json.dumps(checks, indent=2, ensure_ascii=False))
    output = Path(output_root)/stage
    output.mkdir(parents=True, exist_ok=True)
    return PY_STAGES[stage]["runner"](root/stage, output, recompute=recompute, palette=palette)


def export_original_python_sources(output_dir):
    """Export the recovered original analysis/plot scripts without executing them.

    Original scripts retain historical file-path settings as provenance. Inspect
    and configure their input/output paths before running raw GEO reanalyses.
    The integrated plotting stages above do not use those historical paths.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name, source in RECOVERED_ANALYSIS_SOURCES.items():
        (root/name).write_text(source,encoding="utf-8")
    return str(root)


def python_cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list",action="store_true",help="List recovered stages; no scientific dependencies are imported")
    parser.add_argument("--check",action="store_true",help="Check input files and installed dependencies without running analyses")
    parser.add_argument("--stage",choices=["all",*PY_STAGES],default="all")
    parser.add_argument("--data-root",type=Path,default=Path(__file__).resolve().parent/"data"/"python")
    parser.add_argument("--output-root",type=Path,default=Path(__file__).resolve().parent/"results"/"python")
    parser.add_argument("--recompute",action="store_true",help="Recompute Figure 4 model, OOF predictions, bootstrap CI, calibration and DCA")
    parser.add_argument("--palette",choices=["screenshot","source"],default="screenshot")
    parser.add_argument("--export-original-sources",type=Path)
    args = parser.parse_args(argv)
    if args.export_original_sources:
        print(export_original_python_sources(args.export_original_sources)); return
    stages = list(PY_STAGES) if args.stage == "all" else [args.stage]
    if args.list:
        print(json.dumps(list_python_stages(),indent=2,ensure_ascii=False));return
    if args.check:
        print(json.dumps(check_python_inputs(args.data_root,stages),indent=2,ensure_ascii=False));return
    for stage in stages:
        print(f"Running {stage}",flush=True)
        run_python_stage(stage,args.data_root,args.output_root,recompute=args.recompute,palette=args.palette)


# Recovered source snapshots are included for complete scientific provenance.
# These strings are not evaluated by the standard plotting workflow.
RECOVERED_ANALYSIS_SOURCES = {
    'fig04_fit_refit.py': (
        '"""Fit and internally evaluate a fixed four-gene ridge model in GSE38713."""\n'
        'import argparse\n'
        'from hashlib import sha256\n'
        'import json\n'
        'from pathlib import Path\n'
        '\n'
        'import numpy as np\n'
        'import pandas as pd\n'
        'from scipy.special import expit, logit\n'
        'from scipy.stats import mannwhitneyu\n'
        'from sklearn.linear_model import LogisticRegression\n'
        'from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve\n'
        'from sklearn.model_selection import StratifiedKFold\n'
        'from sklearn.pipeline import make_pipeline\n'
        'from sklearn.preprocessing import StandardScaler\n'
        'import statsmodels.api as sm\n'
        'from statsmodels.stats.multitest import multipletests\n'
        'from statsmodels.stats.proportion import proportion_confint\n'
        '\n'
        "GENES = ('CASP1', 'MMP9', 'PECAM1', 'TYROBP')\n"
        'SEED = 20261111\n'
        'FOLDS = 5\n'
        'BOOTSTRAPS = 5000\n'
        '\n'
        '\n'
        'def fit_model(x, y):\n'
        '    model = make_pipeline(StandardScaler(), LogisticRegression(\n'
        "        C=1, solver='lbfgs', max_iter=5000, tol=1e-10))\n"
        '    model.fit(x, y)\n'
        '    scaler, regression = model.steps[0][1], model.steps[1][1]\n'
        '    if regression.n_iter_[0] >= regression.max_iter:\n'
        "        raise RuntimeError('Ridge logistic regression failed to converge')\n"
        '    weights = regression.coef_[0] / scaler.scale_\n'
        '    intercept = float(regression.intercept_[0] - weights @ scaler.mean_)\n'
        '    return model, intercept, weights\n'
        '\n'
        '\n'
        'def cross_validate(x, y, seed):\n'
        '    predictions = np.full(len(y), np.nan)\n'
        '    assignments = np.zeros(len(y), dtype=int)\n'
        '    fits = []\n'
        '    splits = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)\n'
        '    for fold, (train, test) in enumerate(splits.split(x, y), start=1):\n'
        '        model, intercept, weights = fit_model(x[train], y[train])\n'
        '        assert not np.any(assignments[test]) and not np.intersect1d(train, test).size\n'
        '        predictions[test] = model.predict_proba(x[test])[:, 1]\n'
        '        assignments[test] = fold\n'
        '        assert np.allclose(predictions[test], expit(intercept + x[test] @ weights), rtol=0, atol=1e-12)\n'
        "        fits.append({'fold': fold, 'n_train': len(train), 'n_test': len(test),\n"
        "                     'UC_test': int(y[test].sum()), 'control_test': int((y[test] == 0).sum()),\n"
        "                     'intercept_raw_log2': intercept, **dict(zip(GENES, weights))})\n"
        '    assert np.isfinite(predictions).all() and (assignments > 0).all()\n'
        '    return predictions, assignments, fits\n'
        '\n'
        '\n'
        'def auc_interval(y, predictions):\n'
        '    rng = np.random.default_rng(SEED)\n'
        '    cases, controls = predictions[y == 1], predictions[y == 0]\n'
        '    draws = []\n'
        '    for _ in range(BOOTSTRAPS):\n'
        '        pos = rng.choice(cases, len(cases), replace=True)[:, None]\n'
        '        neg = rng.choice(controls, len(controls), replace=True)[None, :]\n'
        '        draws.append(float(np.mean((pos > neg) + .5 * (pos == neg))))\n'
        '    lower, upper = np.quantile(draws, [.025, .975])\n'
        '    return float(lower), float(upper)\n'
        '\n'
        '\n'
        'def calibration_outputs(y, oof, output):\n'
        '    logits = logit(oof)\n'
        '    calibration = sm.GLM(y, sm.add_constant(logits), family=sm.families.Binomial()).fit()\n'
        '    overall = sm.GLM(y, np.ones((len(y), 1)), offset=logits, family=sm.families.Binomial()).fit()\n'
        '    if not calibration.converged or not overall.converged:\n'
        "        raise RuntimeError('Calibration regression failed to converge')\n"
        '    grid = np.linspace(max(.001, oof.min()), min(.999, oof.max()), 300)\n'
        "    pd.DataFrame({'predicted_probability': grid,\n"
        "                  'observed_probability': expit(calibration.params[0] + calibration.params[1] * logit(grid))\n"
        "                  }).to_csv(output / 'calibration_curve.csv', index=False)\n"
        '    bins = []\n'
        "    for key in np.unique(bin_ids := pd.qcut(oof, 3, labels=False, duplicates='drop')):\n"
        '        mask = bin_ids == key\n'
        '        positives, count = int(y[mask].sum()), int(mask.sum())\n'
        "        low, high = proportion_confint(positives, count, method='wilson')\n"
        "        bins.append({'n': count, 'n_uc': positives, 'mean_predicted': oof[mask].mean(),\n"
        "                     'observed_fraction': positives / count, 'ci95_lower': low, 'ci95_upper': high})\n"
        "    pd.DataFrame(bins).to_csv(output / 'calibration_bins.csv', index=False)\n"
        '    intervals = calibration.conf_int()\n'
        "    return {'intercept_joint': float(calibration.params[0]), 'slope': float(calibration.params[1]),\n"
        "            'intercept_joint_ci95': intervals[0].tolist(), 'slope_ci95': intervals[1].tolist(),\n"
        "            'calibration_in_the_large': float(overall.params[0]),\n"
        "            'calibration_in_the_large_ci95': overall.conf_int()[0].tolist()}\n"
        '\n'
        '\n'
        'def decision_curve(y, oof, output):\n'
        '    rows = []\n'
        '    for threshold in np.arange(1, 96) / 100:\n'
        '        odds = threshold / (1 - threshold)\n'
        '        positive = oof >= threshold\n'
        '        tp = np.sum(positive & (y == 1))\n'
        '        fp = np.sum(positive & (y == 0))\n'
        "        rows.append({'threshold': threshold, 'model_net_benefit': (tp - odds * fp) / len(y),\n"
        "                     'treat_all_net_benefit': (y.sum() - odds * (y == 0).sum()) / len(y),\n"
        "                     'treat_none_net_benefit': 0.})\n"
        "    pd.DataFrame(rows).to_csv(output / 'decision_curve_oof.csv', index=False)\n"
        '\n'
        '\n'
        'def expression_statistics(data, output):\n'
        '    rows = []\n'
        '    for gene in GENES:\n'
        "        control = data.loc[data.group.eq('Control'), gene]\n"
        "        uc = data.loc[data.group.eq('UC'), gene]\n"
        "        result = mannwhitneyu(uc, control, alternative='two-sided', method='auto')\n"
        "        rows.append({'gene': gene, 'control_n': len(control), 'uc_n': len(uc),\n"
        "                     'control_median_log2': control.median(), 'uc_median_log2': uc.median(),\n"
        "                     'U_statistic': result.statistic, 'p_value': result.pvalue})\n"
        '    df = pd.DataFrame(rows)\n'
        "    df['q_bh'] = multipletests(df.p_value, method='fdr_bh')[1]\n"
        "    df.to_csv(output / 'expression_statistics.csv', index=False)\n"
        '\n'
        '\n'
        'def run(inputs, output):\n'
        '    data = pd.read_csv(inputs)\n'
        '    assert data.sample_id.is_unique and data.donor_id.is_unique\n'
        "    assert data.group.value_counts().to_dict() == {'UC': 23, 'Control': 13}\n"
        "    assert data.activity.value_counts().to_dict() == {'Active UC': 15, 'Control': 13, 'Inactive UC': 8}\n"
        '    x = data[list(GENES)].to_numpy(dtype=float)\n'
        '    y = data.UC_status.to_numpy(dtype=int)\n'
        "    assert np.isfinite(x).all() and np.array_equal(y, data.group.eq('UC').astype(int).to_numpy())\n"
        '\n'
        '    output.mkdir(parents=True, exist_ok=True)\n'
        '    oof, folds, fold_fits = cross_validate(x, y, SEED)\n'
        '    full_model, intercept, weights = fit_model(x, y)\n'
        '    full_predictions = full_model.predict_proba(x)[:, 1]\n'
        '    formula_error = float(np.max(np.abs(full_predictions - expit(intercept + x @ weights))))\n'
        '    assert formula_error < 1e-12\n'
        '    data.assign(fold_id=folds, p_oof=oof, p_full_fit=full_predictions).to_csv(\n'
        "        output / 'sample_predictions.csv', index=False, float_format='%.15g')\n"
        "    pd.DataFrame(fold_fits).to_csv(output / 'fold_models.csv', index=False)\n"
        '    scaler, logistic = full_model.steps[0][1], full_model.steps[1][1]\n'
        "    pd.DataFrame({'term': ['Intercept', *GENES], 'coefficient_raw_log2': [intercept, *weights],\n"
        "                  'coefficient_standardized': [logistic.intercept_[0], *logistic.coef_[0]],\n"
        "                  'training_mean': ['NA', *scaler.mean_], 'training_sd_ddof0': ['NA', *scaler.scale_]\n"
        "                  }).to_csv(output / 'full_model_coefficients.csv', index=False)\n"
        '\n'
        '    lower, upper = auc_interval(y, oof)\n'
        '    roc_x, roc_y, thresholds = roc_curve(y, oof, drop_intermediate=False)\n'
        "    pd.DataFrame({'fpr': roc_x, 'tpr': roc_y, 'threshold': thresholds}).to_csv(output / 'roc_oof.csv', index=False)\n"
        '    calibration = calibration_outputs(y, oof, output)\n'
        '    decision_curve(y, oof, output)\n'
        '    expression_statistics(data, output)\n'
        '    tp = int(((oof >= .5) & (y == 1)).sum())\n'
        '    tn = int(((oof < .5) & (y == 0)).sum())\n'
        "    performance = {'oof': {'auc': float(roc_auc_score(y, oof)), 'ci95_lower': lower, 'ci95_upper': upper,\n"
        "                           'brier_score': float(brier_score_loss(y, oof)),\n"
        "                           'cutoff': .5, 'sensitivity': tp / 23, 'specificity': tn / 13,\n"
        "                           'tp': tp, 'tn': tn, 'fp': 13-tn, 'fn': 23-tp},\n"
        "                   'full_fit': {'auc': float(roc_auc_score(y, full_predictions)),\n"
        "                                'brier_score': float(brier_score_loss(y, full_predictions))}}\n"
        "    report = {'dataset': 'GSE38713', 'n_uc': 23, 'n_control': 13, 'activity': {'active_uc': 15, 'remission_uc': 8},\n"
        "              'input_sha256': sha256(inputs.read_bytes()).hexdigest(),\n"
        "              'input_scale': 'GEO-deposited GPL570 GC-RMA log2, mean across unambiguous probes',\n"
        "              'analysis': 'Single-cohort model development and fixed-panel internal cross-validation',\n"
        "              'model': {'genes': list(GENES), 'method': 'StandardScaler and L2 logistic regression (C=1)',\n"
        "                        'raw_log2_intercept': intercept, 'raw_log2_coefficients': dict(zip(GENES, weights)),\n"
        "                        'full_fit_formula_max_abs_error': formula_error},\n"
        "              'validation': {'random_seed': SEED, 'n_stratified_folds': FOLDS,\n"
        "                             'scaling_and_estimation_within_training_fold': True,\n"
        "                             'auc_interval': f'{BOOTSTRAPS} stratified bootstrap resamples of fixed OOF predictions'},\n"
        "              'performance': performance, 'calibration': calibration}\n"
        "    (output / 'model_summary.json').write_text(json.dumps(report, indent=2) + '\\n', encoding='utf-8')\n"
        "    print(json.dumps({'oof': performance['oof'], 'calibration': calibration,\n"
        "                      'intercept': intercept, 'coefficients': dict(zip(GENES, weights))}, indent=2))\n"
        '\n'
        '\n'
        "if __name__ == '__main__':\n"
        '    cli = argparse.ArgumentParser(description=__doc__)\n'
        "    cli.add_argument('--input', type=Path, required=True)\n"
        "    cli.add_argument('--output', type=Path, required=True)\n"
        '    args = cli.parse_args()\n'
        '    run(args.input, args.output)\n'
    ),
    'fig04_plot_refit.py': (
        '"""Draw the five-panel scientific figure for the GSE38713 four-gene model."""\n'
        '\n'
        'import argparse\n'
        'import json\n'
        'from pathlib import Path\n'
        '\n'
        'import matplotlib\n'
        "matplotlib.use('Agg')\n"
        'import matplotlib.pyplot as plt\n'
        'from matplotlib.lines import Line2D\n'
        'from matplotlib.ticker import FixedLocator, FormatStrFormatter, MultipleLocator\n'
        'import numpy as np\n'
        'import pandas as pd\n'
        'from scipy.special import expit, logit\n'
        '\n'
        '\n'
        "GENES = ('CASP1', 'MMP9', 'PECAM1', 'TYROBP')\n"
        "BLUE = '#14297A'\n"
        "RED = '#8C0026'\n"
        "INK = '#151920'\n"
        "MUTED = '#536577'\n"
        "IDEAL = '#677B8C'\n"
        "GENE_COLORS = {'CASP1': '#267044', 'MMP9': RED, 'PECAM1': BLUE, 'TYROBP': '#643887'}\n"
        '\n'
        '\n'
        'def panel_header(ax, letter, title, subtitle, top=False):\n'
        '    title_y, subtitle_y = (1.12, 1.056) if top else (1.18, 1.10)\n'
        '    ax.text(-.15 if top else -.17, title_y, letter, transform=ax.transAxes,\n'
        "            fontfamily='Times New Roman', fontsize=29, va='center')\n"
        '    ax.text(0, title_y, title, transform=ax.transAxes, fontsize=15,\n'
        "            fontweight='bold', va='center')\n"
        '    ax.text(0, subtitle_y, subtitle, transform=ax.transAxes, fontsize=9.5,\n'
        "            color=MUTED, va='center')\n"
        '\n'
        '\n'
        'def expression_panel(ax, data, stats):\n'
        "    stats = stats.set_index('gene')\n"
        "    offsets = {'Control': -.17, 'UC': .17}\n"
        '    rng = np.random.default_rng(20261111)\n'
        '    bottom = float(data[list(GENES)].min().min())\n'
        '    top = float(data[list(GENES)].max().max())\n'
        '    span = top - bottom\n'
        '    for i, gene in enumerate(GENES):\n'
        "        for group, color in [('Control', BLUE), ('UC', RED)]:\n"
        "            values = data.loc[data['group'].eq(group), gene].to_numpy()\n"
        '            pos = i + offsets[group]\n'
        '            ax.boxplot(values, positions=[pos], widths=.25, patch_artist=True,\n'
        '                       manage_ticks=False, showfliers=False,\n'
        "                       boxprops={'facecolor': color, 'edgecolor': color, 'lw': 1.2},\n"
        "                       whiskerprops={'color': color, 'lw': 1.2},\n"
        "                       capprops={'color': color, 'lw': 1.2},\n"
        "                       medianprops={'color': 'white', 'lw': 1.7})\n"
        '            ax.scatter(pos + rng.uniform(-.11, .11, len(values)), values,\n'
        "                       s=19, facecolor=color, edgecolor='white', lw=.45,\n"
        '                       alpha=.91, zorder=3)\n'
        "        q = float(stats.loc[gene, 'q_bh'])\n"
        "        sig = '***' if q < .001 else '**' if q < .01 else '*' if q < .05 else 'ns'\n"
        '        y = top + .038 * span\n'
        "        lo, hi = i + offsets['Control'], i + offsets['UC']\n"
        '        ax.plot([lo, lo, hi, hi], [y, y+.018*span, y+.018*span, y], color=INK, lw=1)\n'
        "        ax.text(i, y+.028*span, sig, ha='center', va='bottom', fontsize=10, fontweight='bold')\n"
        '    ax.set(xlim=(-.54, 3.54), ylim=(bottom-.055*span, top+.155*span),\n'
        "           ylabel='GC-RMA log2 expression', xticks=range(4), xticklabels=GENES)\n"
        "    ax.tick_params(axis='x', labelsize=10)\n"
        '    ax.yaxis.set_major_locator(MultipleLocator(2))\n'
        '    for tick in ax.get_xticklabels():\n'
        "        tick.set_fontweight('bold')\n"
        "    handles = [Line2D([], [], marker='s', ms=7, markerfacecolor=color,\n"
        '                      markeredgecolor=color, lw=0, label=group)\n'
        "               for group, color in [('Control', BLUE), ('UC', RED)]]\n"
        "    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1, 1.145),\n"
        '              ncols=2, frameon=False, columnspacing=1.0, handletextpad=.2,\n'
        '              borderaxespad=0, fontsize=9.5)\n'
        "    ax.text(0, -.15, 'BH-adjusted P: * < 0.05, ** < 0.01, *** < 0.001',\n"
        "            transform=ax.transAxes, color=MUTED, fontsize=8.5, va='top')\n"
        '\n'
        '\n'
        'def display_number(value):\n'
        "    return f'{value:.3g}'\n"
        '\n'
        '\n'
        'def gene_ticks(lo, hi, points, extent):\n'
        '    if extent < 12:\n'
        '        return np.array([])\n'
        '    if extent < 30:\n'
        '        mid = round(lo + (hi-lo)/2, 1)\n'
        '        return np.array([lo, mid, hi])\n'
        '    step = next(s for s in (.5, 1, 2, 2.5, 5, 10, 20)\n'
        '                if (hi - lo) / s <= max(1, extent / 8))\n'
        '    interior = np.arange(np.ceil(lo / step) * step, hi + 1e-10, step)\n'
        '    min_spacing = 5.2\n'
        '    keep = [v for v in interior\n'
        '            if min(abs(points(v) - points(lo)), abs(points(v) - points(hi))) >= min_spacing]\n'
        '    return np.unique(np.r_[lo, keep, hi])\n'
        '\n'
        '\n'
        'def nomogram_panel(ax, data, model):\n'
        "    coefficients = model['raw_log2_coefficients']\n"
        "    intercept = float(model['raw_log2_intercept'])\n"
        '    ranges = {g: (np.floor(data[g].min()*2)/2, np.ceil(data[g].max()*2)/2) for g in GENES}\n'
        '    minimums = {g: min(coefficients[g] * endpoints[0], coefficients[g] * endpoints[1])\n'
        '                for g, endpoints in ranges.items()}\n'
        '    spans = {g: abs(coefficients[g]) * (hi - lo) for g, (lo, hi) in ranges.items()}\n'
        '    scale = 100 / max(spans.values())\n'
        '    total_max = sum(spans.values()) * scale\n'
        '    min_lp = intercept + sum(minimums.values())\n'
        '    total_points = sum(scale*(coefficients[g]*data[g].to_numpy() - minimums[g]) for g in GENES)\n'
        "    np.testing.assert_allclose(expit(min_lp + total_points/scale), data['p_full_fit'], atol=1e-10, rtol=0)\n"
        '\n'
        '    left, right = .24, .99\n'
        '    ax.set(xlim=(0, 1), ylim=(.07, 7.13))\n'
        '    ax.set_axis_off()\n'
        '\n'
        '    def line(label, y, domain, values, point_positions, *, extent=None,\n'
        '             color=BLUE, bold=False, italic=False, explanation=None):\n'
        '        x = left + (right-left)*np.asarray(point_positions)/domain\n'
        '        line_end = max(x) if extent is None else left + (right-left)*extent/domain\n'
        '        line_start = min(x) if extent is None else left\n'
        "        ax.text(0, y, label, ha='left', va='center', fontsize=9.8,\n"
        "                fontweight='bold' if bold else 'normal',\n"
        "                fontstyle='italic' if italic else 'normal')\n"
        "        ax.plot([line_start, line_end], [y, y], lw=1.75, color=color, solid_capstyle='butt')\n"
        '        ax.vlines(x, y-.115, y, lw=1, color=color)\n'
        '        for position, value in zip(x, values):\n'
        "            ax.text(position, y-.185, display_number(value), ha='center', va='top', fontsize=8.3)\n"
        '        if explanation:\n'
        "            ax.text(line_end+.022, y, explanation, ha='left', va='center', fontsize=8.2, color=MUTED)\n"
        '\n'
        '    point_ticks = np.arange(0, 101, 10)\n'
        "    line('Points', 6.68, 100, point_ticks, point_ticks, bold=True)\n"
        "    for gene, y in zip(('MMP9', 'PECAM1', 'CASP1', 'TYROBP'), (5.76, 4.84, 3.92, 3.0)):\n"
        '        lo, hi = ranges[gene]\n'
        '        def points(values):\n'
        '            return scale*(coefficients[gene]*values - minimums[gene])\n'
        '        extent = scale*spans[gene]\n'
        '        values = gene_ticks(lo, hi, points, extent)\n'
        '        explanation = None\n'
        '        if not len(values):\n'
        '            first, last = (lo, hi) if coefficients[gene] >= 0 else (hi, lo)\n'
        "            explanation = f'{first:g} to {last:g} (left to right)'\n"
        '        line(gene, y, 100, values, points(values), extent=extent,\n'
        '             color=GENE_COLORS[gene], italic=True, explanation=explanation)\n'
        '\n'
        '    ticks = np.arange(0, total_max + 1e-10, 25)\n'
        '    if total_max - ticks[-1] >= .053 * total_max:\n'
        '        ticks = np.r_[ticks, total_max]\n'
        "    line('Total points', 1.91, total_max, ticks, ticks, extent=total_max, bold=True)\n"
        '    probs = np.array([.001, .01, .05, .1, .2, .3, .5, .7, .8, .9, .95, .99, .999])\n'
        '    probs_points = scale * (logit(probs) - min_lp)\n'
        '    # Choose labeled probabilities separated in display space, prioritizing the middle of the range.\n'
        '    chosen = []\n'
        '    for i in (6, 3, 9, 1, 11, 5, 7, 0, 12, 2, 10, 4, 8):\n'
        '        point = probs_points[i]\n'
        '        if 0 <= point <= total_max and all(abs(point - probs_points[j]) >= .063*total_max for j in chosen):\n'
        '            chosen.append(i)\n'
        '    chosen.sort()\n'
        '    if len(chosen) < 2:\n'
        '        probs_points = np.linspace(0, total_max, 3)\n'
        '        probs = expit(min_lp + probs_points/scale)\n'
        '        chosen = [0, 1, 2]\n'
        "    line('Predicted UC\\nprobability', .76, total_max, probs[chosen], probs_points[chosen],\n"
        '         color=RED, bold=True)\n'
        '\n'
        '\n'
        'def roc_panel(ax, roc, metrics):\n'
        '    ax.plot([0, 1], [0, 1], color=IDEAL, ls=(0, (4, 3)), lw=1.2)\n'
        "    ax.plot(roc['fpr'], roc['tpr'], color=BLUE, lw=2.4, drawstyle='steps-post')\n"
        '    ax.text(.97, .105, f"AUC = {metrics[\'auc\']:.3f}", transform=ax.transAxes,\n'
        "            ha='right', va='bottom', fontsize=12)\n"
        '    ax.text(.97, .034, f"95% CI {metrics[\'ci95_lower\']:.3f}-{metrics[\'ci95_upper\']:.3f}",\n'
        "            transform=ax.transAxes, ha='right', va='bottom', fontsize=10)\n"
        "    ax.set(xlim=(-.01, 1.01), ylim=(-.01, 1.04), xlabel='1 - specificity', ylabel='Sensitivity')\n"
        '    ax.xaxis.set_major_locator(MultipleLocator(.2))\n'
        '    ax.yaxis.set_major_locator(MultipleLocator(.2))\n'
        '\n'
        '\n'
        'def calibration_panel(ax, bins, curve, calibration, metrics):\n'
        "    ax.plot([0, 1], [0, 1], color=IDEAL, ls=(0, (4, 3)), lw=1.2, label='Ideal calibration')\n"
        "    if calibration.get('slope') is not None:\n"
        "        ax.plot(curve['predicted_probability'], curve['observed_probability'],\n"
        "                color=BLUE, lw=2.2, label='Logistic calibration')\n"
        "    errors = np.maximum(0, np.vstack([bins['observed_fraction'] - bins['ci95_lower'],\n"
        "                                      bins['ci95_upper'] - bins['observed_fraction']]))\n"
        "    ax.errorbar(bins['mean_predicted'], bins['observed_fraction'], yerr=errors,\n"
        "                fmt='o', color=RED, lw=1.35, capsize=3, ms=5, label='Risk groups (95% CI)')\n"
        '    slope_label = f"{calibration[\'slope\']:.3f}" if calibration.get(\'slope\') is not None else \'NA\'\n'
        '    ax.text(.04, .966, f"Slope = {slope_label}\\nBrier = {metrics[\'brier_score\']:.3f}",\n'
        "            transform=ax.transAxes, ha='left', va='top', fontsize=10.5, linespacing=1.35)\n"
        "    ax.legend(loc='lower right', frameon=False, fontsize=8.1, handlelength=1.7, borderaxespad=.45)\n"
        "    ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.04), xlabel='Predicted probability',\n"
        "           ylabel='Observed UC proportion')\n"
        '    ax.xaxis.set_major_locator(MultipleLocator(.2))\n'
        '    ax.yaxis.set_major_locator(MultipleLocator(.2))\n'
        '\n'
        '\n'
        'def dca_panel(ax, curve):\n'
        "    ax.plot(curve['threshold'], curve['model_net_benefit'], lw=2.4, color=BLUE, label='Four-gene model')\n"
        "    ax.plot(curve['threshold'], curve['treat_all_net_benefit'], lw=1.8, color=RED, label='Treat all')\n"
        "    ax.plot(curve['threshold'], curve['treat_none_net_benefit'], lw=1.1, color=INK,\n"
        "            ls=(0, (4, 3)), label='Treat none')\n"
        "    maximum = max(curve['model_net_benefit'].max(), curve['treat_all_net_benefit'].max())\n"
        '    upper = np.ceil((maximum + .03) * 10) / 10\n'
        "    ax.set(xlim=(0, .952), ylim=(-.065, upper), xlabel='Threshold probability', ylabel='Net benefit')\n"
        '    ax.xaxis.set_major_locator(FixedLocator([0, .2, .4, .6, .8, .95]))\n'
        "    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))\n"
        '    ax.yaxis.set_major_locator(MultipleLocator(.2))\n'
        "    ax.legend(loc='lower left', bbox_to_anchor=(.01, .07), frameon=True, edgecolor='none',\n"
        '              framealpha=.91, fontsize=8.4, handlelength=1.7)\n'
        '\n'
        '\n'
        'def main():\n'
        '    parser = argparse.ArgumentParser(description=__doc__)\n'
        "    parser.add_argument('--results', required=True, type=Path)\n"
        '    args = parser.parse_args()\n'
        '    results = args.results.resolve()\n'
        "    figure_dir = results / 'figures'\n"
        '    figure_dir.mkdir(parents=True, exist_ok=True)\n'
        "    summary = json.loads((results / 'model_summary.json').read_text(encoding='utf-8'))\n"
        "    data = pd.read_csv(results / 'sample_predictions.csv')\n"
        "    counts = data['group'].value_counts()\n"
        "    assert counts.to_dict() == {'UC': 23, 'Control': 13}\n"
        '    cohort = f"GSE38713  |  UC n={counts[\'UC\']}, control n={counts[\'Control\']}"\n'
        "    metrics = summary['performance']['oof']\n"
        '\n'
        "    mpl_style = {'font.family': 'Arial', 'font.size': 10, 'pdf.fonttype': 42,\n"
        "                 'axes.linewidth': 1.1, 'axes.spines.top': False, 'axes.spines.right': False,\n"
        "                 'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': INK,\n"
        "                 'xtick.color': INK, 'ytick.color': INK, 'axes.labelsize': 11,\n"
        "                 'xtick.labelsize': 9.8, 'ytick.labelsize': 9.8,\n"
        "                 'xtick.major.size': 4, 'ytick.major.size': 4, 'savefig.facecolor': 'white'}\n"
        '    with plt.rc_context(mpl_style):\n'
        '        figure = plt.figure(figsize=(13.6, 9.4))\n'
        '        root = figure.add_gridspec(nrows=2, ncols=1, height_ratios=[1.10, 1],\n'
        '                                  left=.077, right=.965, top=.876, bottom=.108, hspace=.53)\n'
        '        top = root[0].subgridspec(1, 2, width_ratios=[.96, 1.04], wspace=.28)\n'
        '        bottom = root[1].subgridspec(1, 3, wspace=.38)\n'
        '        a, b = [figure.add_subplot(p) for p in top]\n'
        '        c, d, e = [figure.add_subplot(p) for p in bottom]\n'
        "        panel_header(a, 'A', 'Gene expression', cohort, top=True)\n"
        "        expression_panel(a, data, pd.read_csv(results / 'expression_statistics.csv'))\n"
        "        panel_header(b, 'B', 'Four-gene nomogram', cohort + '  |  Full-cohort refit', top=True)\n"
        "        nomogram_panel(b, data, summary['model'])\n"
        "        panel_header(c, 'C', 'ROC curve', cohort)\n"
        "        roc_panel(c, pd.read_csv(results / 'roc_oof.csv'), metrics)\n"
        "        panel_header(d, 'D', 'Calibration', cohort)\n"
        "        calibration_panel(d, pd.read_csv(results / 'calibration_bins.csv'),\n"
        "                          pd.read_csv(results / 'calibration_curve.csv'), summary['calibration'], metrics)\n"
        "        panel_header(e, 'E', 'Decision curve', cohort)\n"
        "        dca_panel(e, pd.read_csv(results / 'decision_curve_oof.csv'))\n"
        "        notes = ['Five-fold out-of-fold predictions', 'Five-fold out-of-fold predictions',\n"
        "                 'Five-fold out-of-fold predictions (exploratory)']\n"
        '        for axis, note in zip([c,d,e], notes):\n'
        "            axis.text(0, -.23, note, transform=axis.transAxes, va='top', fontsize=8.6, color=MUTED)\n"
        '\n'
        "        stem = figure_dir / 'GSE38713_four_gene_refit'\n"
        "        figure.savefig(stem.with_suffix('.pdf'))\n"
        "        figure.savefig(stem.with_suffix('.png'), dpi=600)\n"
        '        plt.close(figure)\n'
        '\n'
        '\n'
        "if __name__ == '__main__':\n"
        '    main()\n'
    ),
    'fig06_reference_violins.py': (
        'from pathlib import Path\n'
        'import numpy as np\n'
        'import pandas as pd\n'
        'import matplotlib\n'
        'matplotlib.use("Agg")\n'
        'import matplotlib.pyplot as plt\n'
        'from matplotlib.patches import Rectangle\n'
        'from matplotlib.ticker import MaxNLocator\n'
        'from scipy.stats import mannwhitneyu, false_discovery_control, gaussian_kde\n'
        '\n'
        'BASE = Path(__file__).resolve().parents[1]\n'
        'DATA = Path.home() / "Desktop" / "TYROBP_疾病对照分析_2026-11-18" / "tables"\n'
        'OUT = BASE / "outputs" / "TYROBP_参考图版"\n'
        'OUT.mkdir(parents=True, exist_ok=True)\n'
        'GENES = ("TYROBP", "CASP1", "MMP9", "PECAM1")\n'
        'INK = "#1F2A33"\n'
        'BLUE = "#153D61"\n'
        'RED = "#A4213F"\n'
        'plt.rcParams.update({\n'
        '    "font.family": "Arial", "font.size": 11, "mathtext.fontset": "stixsans",\n'
        '    "text.color": INK, "axes.edgecolor": INK, "axes.labelcolor": INK,\n'
        '    "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 1.05,\n'
        '    "axes.spines.top": False, "axes.spines.right": False,\n'
        '    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",\n'
        '})\n'
        '\n'
        '\n'
        'def draw_group(ax, center, values, color, bandwidth, seed, mark_scale=1):\n'
        '    y = np.linspace(values.min(), values.max(), 400)\n'
        '    kde = gaussian_kde(values, bw_method=bandwidth / values.std(ddof=1))\n'
        '    density = kde(y)\n'
        '    widths = .37 * density / density.max()\n'
        '    ax.fill_betweenx(y, center-widths, center+widths, facecolor=color,\n'
        '                     edgecolor=color, linewidth=.65, zorder=1)\n'
        '    # Tukey box and whiskers, with every observation drawn separately below.\n'
        '    q1, med, q3 = np.quantile(values, [.25, .5, .75])\n'
        '    iqr = q3-q1\n'
        '    whisk_lo = values[values >= q1-1.5*iqr].min()\n'
        '    whisk_hi = values[values <= q3+1.5*iqr].max()\n'
        '    ax.vlines(center, whisk_lo, whisk_hi, colors=INK, linewidth=.75, zorder=2)\n'
        '    w = .115\n'
        '    ax.add_patch(Rectangle((center-w/2, q1), w, iqr, facecolor="white",\n'
        '                           edgecolor=INK, linewidth=.9, zorder=3))\n'
        '    ax.hlines(med, center-w/2, center+w/2, color=INK, linewidth=1.15, zorder=4)\n'
        '    # Choose small reproducible horizontal offsets, avoiding neighbouring dots.\n'
        '    # The measured expression is always the unshifted vertical coordinate.\n'
        '    rng = np.random.default_rng(seed)\n'
        '    placed = []\n'
        '    trans = ax.transData\n'
        '    for i in np.argsort(values, kind="stable"):\n'
        '        value = values[i]\n'
        '        max_x = min(.205, float(np.interp(value, y, widths))*.91)\n'
        '        start = rng.uniform(-max_x,max_x)\n'
        '        options = np.r_[start,np.linspace(-max_x,max_x,73)]\n'
        '        xy = trans.transform(np.c_[center+options,np.full(options.shape,value)])\n'
        '        if placed:\n'
        '            close = np.sqrt(((xy[:,None,:]-np.array(placed)[None,:,:])**2).sum(axis=2)).min(axis=1)\n'
        '        else:\n'
        '            close = np.full(len(options), 1e5)\n'
        '        spacing = 5.9 * mark_scale * ax.figure.dpi/72\n'
        '        # Encourage a clear view of the box when a sample sits over its interior.\n'
        '        pref = np.abs(options-start) + (.065 * (np.abs(options)<.065) if q1<=value<=q3 else 0)\n'
        '        viable = np.flatnonzero(close >= spacing)\n'
        '        choice = viable[np.argmin(pref[viable])] if len(viable) else np.argmax(close-pref*.01)\n'
        '        placed.append(xy[choice])\n'
        '        ax.scatter(center+options[choice],value,s=25*mark_scale**2,facecolors=color,edgecolors="white",\n'
        '                   linewidths=.78*mark_scale,zorder=5)\n'
        '\n'
        '\n'
        'def format_panel(ax, gene, comp, pair, adjusted, p_symbol):\n'
        '    lo = min(a.min() for a in pair)\n'
        '    hi = max(a.max() for a in pair)\n'
        '    span = hi-lo\n'
        '    lower, upper = lo - .145*span, hi + .36*span\n'
        '    ax.set(xlim=(-.55,1.55), ylim=(lower,upper))\n'
        '    ticks = MaxNLocator(nbins=5,steps=[1,2,2.5,5,10]).tick_values(lower, hi + .1*span)\n'
        '    ticks = ticks[(ticks >= lower) & (ticks <= hi + .12*span)]\n'
        '    ax.set_yticks(ticks)\n'
        '    ax.set_yticklabels([f"{x:.1f}" if (len(ticks)>1 and (ticks[1]-ticks[0])<1) else f"{x:g}" for x in ticks])\n'
        '    labels = ("CD", "UC") if comp=="CDc" else ("Control", "UC")\n'
        '    ax.set_xticks([0,1], [f"{name}\\n(n = {len(a)})" for name,a in zip(labels,pair)])\n'
        '    ax.tick_params(axis="x", length=0, pad=4, labelsize=10.7)\n'
        '    ax.tick_params(axis="y", length=3.5, width=.9, pad=3.5, labelsize=10)\n'
        '    ax.set_ylabel(r"Gene expression (log$_2$ scale)", labelpad=6, fontsize=10.7)\n'
        '    ax.set_title(gene, fontsize=14.7, fontstyle="italic", pad=9)\n'
        '    bracket = hi+.175*span\n'
        '    ax.plot([0,0,1,1],[bracket-.040*span,bracket,bracket,bracket-.040*span],\n'
        '            lw=.85, color=INK, solid_capstyle="butt")\n'
        '    label = p_symbol + (f" = {adjusted:.3f}" if adjusted>=.001 else " < 0.0001")\n'
        '    ax.text(.5,bracket+.032*span,label,ha="center",va="bottom",fontsize=11.1)\n'
        '\n'
        '\n'
        'def plot(samples, comp, stem, p_symbol=r"$P_{\\mathrm{adj}}$"):\n'
        '    first = samples[samples.group==comp]\n'
        '    uc = samples[samples.group=="UC"]\n'
        '    pairs = [(first[g].to_numpy(),uc[g].to_numpy()) for g in GENES]\n'
        '    raw = [mannwhitneyu(a,b,alternative="two-sided",method="auto").pvalue for a,b in pairs]\n'
        '    adjusted = false_discovery_control(raw, method="bh")\n'
        '    if comp=="CDc":\n'
        '        reference = pd.read_csv(DATA/"four_gene_expression_UC_vs_CDc.csv").set_index("gene")\n'
        '        np.testing.assert_allclose(adjusted,reference.loc[list(GENES),"BH_adjusted_p"])\n'
        '    elif comp=="CO":\n'
        '        assert np.all(adjusted<.0001)\n'
        '    fig, axes = plt.subplots(2,2,figsize=(8.2,7.7))\n'
        '    fig.subplots_adjust(left=.104,right=.974, bottom=.080,top=.94, hspace=.39,wspace=.32)\n'
        '    for ax,gene,pair,p in zip(axes.flat,GENES,pairs,adjusted):\n'
        '        format_panel(ax,gene,comp,pair,p,p_symbol)\n'
        '    fig.canvas.draw()\n'
        '    for ix,(ax,gene,pair) in enumerate(zip(axes.flat,GENES,pairs)):\n'
        '        groups = [part[gene].to_numpy() for _,part in samples.groupby("group")]\n'
        '        centered = np.concatenate([v-v.mean() for v in groups])\n'
        '        bandwidth = np.sqrt((centered**2).sum()/(len(centered)-len(groups))) * np.median([len(v) for v in groups])**(-.2)\n'
        '        for j,(values,col) in enumerate(zip(pair,[BLUE,RED])):\n'
        '            draw_group(ax,j,values,col,bandwidth,seed=136+ix*11+j)\n'
        '    fig.savefig(OUT/(stem+".png"),dpi=450)\n'
        '    fig.savefig(OUT/(stem+".pdf"))\n'
        '    plt.close(fig)\n'
        '\n'
        '\n'
        'if __name__=="__main__":\n'
        '    samples=pd.read_csv(DATA/"selected_samples_expression_predictions.csv")\n'
        '    assert samples.patient_id.is_unique\n'
        '    assert samples.groupby("group").size().to_dict()=={"CO":6,"CDc":19,"UC":24}\n'
        '    plot(samples,"CDc","01_CDc_vs_UC")\n'
        '    plot(samples,"CO","02_Control_vs_UC")\n'
    ),
    'fig06_rev_model_figures.py': (
        '"""Render and audit separate disease-control performances from saved held-out scores."""\n'
        'from pathlib import Path\n'
        '\n'
        'import matplotlib\n'
        'matplotlib.use("Agg")\n'
        'from matplotlib import pyplot as plt\n'
        'from matplotlib.colors import to_rgb\n'
        'from matplotlib.patches import Rectangle\n'
        'from matplotlib.ticker import FuncFormatter\n'
        'import numpy as np\n'
        'import pandas as pd\n'
        'from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve\n'
        'from sklearn.model_selection import StratifiedKFold\n'
        '\n'
        '\n'
        'ROOT = Path.home() / "Desktop" / "TYROBP_疾病对照分析_2026-11-18"\n'
        'DEST = Path(__file__).resolve().parents[1] / "outputs" / "TYROBP_小提琴图与模型性能"\n'
        'INK, MUTED, GRID = "#26333A", "#535F64", "#DCE2E2"\n'
        'plt.rcParams.update({\n'
        '    "font.family": "Arial", "font.size": 10, "text.color": INK,\n'
        '    "axes.labelcolor": INK, "axes.edgecolor": "#899497", "axes.linewidth": .65,\n'
        '    "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": "white",\n'
        '    "axes.facecolor": "white", "savefig.facecolor": "white",\n'
        '    "pdf.fonttype": 42, "ps.fonttype": 42,\n'
        '})\n'
        '\n'
        '\n'
        'def check_results(cohort, result):\n'
        '    name = result.name\n'
        '    y = cohort["UC_positive"].to_numpy(dtype=int)\n'
        '    score = cohort[name + "_score"].to_numpy()\n'
        '    probability = cohort[name + "_probability"].to_numpy()\n'
        '    np.testing.assert_allclose(1 / (1 + np.exp(-score)), probability)\n'
        '    predicted = probability >= result.threshold\n'
        '    counts = confusion_matrix(y, predicted, labels=[1, 0])\n'
        '    tp, fn, fp, tn = counts.ravel()\n'
        '    observed_auc = roc_auc_score(y, score)\n'
        '    np.testing.assert_array_equal([tp, fn, fp, tn], result[["TP", "FN", "FP", "TN"]])\n'
        '    sens = tp / (tp + fn)\n'
        '    spec = tn / (tn + fp)\n'
        '    np.testing.assert_allclose([sens, spec, .5 * (sens + spec), observed_auc],\n'
        '        result[["sensitivity", "specificity", "balanced_accuracy", "auc"]].to_numpy(float))\n'
        '    pos, neg = score[y == 1], score[y == 0]\n'
        '    rng = np.random.default_rng(20261118)\n'
        '    pp = rng.choice(pos, (5000, len(pos)))\n'
        '    nn = rng.choice(neg, (5000, len(neg)))\n'
        '    cmp = pp[:, :, None] - nn[:, None, :]\n'
        '    auc_ci = np.quantile(((cmp > 0) + .5*(cmp == 0)).mean(axis=(1, 2)), [.025, .975])\n'
        '    np.testing.assert_allclose(auc_ci, [result.ci_low, result.ci_high], atol=1e-12)\n'
        '    fpr, tpr, _ = roc_curve(y, score, drop_intermediate=False)\n'
        '    assert np.any(np.isclose(fpr, 1-spec) & np.isclose(tpr, sens))\n'
        '    return counts, fpr, tpr\n'
        '\n'
        '\n'
        'def roc_panel(fig, result, fpr, tpr, color):\n'
        '    ax = fig.add_axes([.13, .212, .34, .34 * 8.8 / 5.15])\n'
        '    ax.plot([0, 1], [0, 1], color="#9AA4A6", ls=(0, (2.5, 3)), lw=1)\n'
        '    ax.plot(fpr, tpr, lw=2.15, color=color, solid_capstyle="butt")\n'
        '    ax.plot(1-result.specificity, result.sensitivity, marker="D", markersize=7,\n'
        '            color=color, markeredgecolor="white", markeredgewidth=1.1, clip_on=False)\n'
        '    ax.set(xlim=(-.012, 1.012), ylim=(-.012, 1.012),\n'
        '           xlabel="False positive rate (1 − specificity)", ylabel="True positive rate (sensitivity)")\n'
        '    ax.spines[["top", "right"]].set_visible(False)\n'
        '    ax.spines["left"].set_bounds(0, 1)\n'
        '    ax.spines["bottom"].set_bounds(0, 1)\n'
        '    ax.set_xticks(np.linspace(0, 1, 6))\n'
        '    ax.set_yticks(np.linspace(0, 1, 6))\n'
        '    for axis in (ax.xaxis, ax.yaxis):\n'
        '        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))\n'
        '    ax.tick_params(length=3, width=.65, labelsize=9)\n'
        '    ax.set_axisbelow(True)\n'
        '    ax.grid(False)\n'
        '    ax.xaxis.labelpad = 7\n'
        '    ax.yaxis.labelpad = 7\n'
        '    ax.text(.042, .909, f"AUC {result.auc:.3f}", va="center", fontsize=15,\n'
        '            weight="bold", color=color, transform=ax.transAxes)\n'
        '    ax.text(.042, .818, f"95% CI  {result.ci_low:.3f}–{result.ci_high:.3f}", va="center",\n'
        '            fontsize=9.75, transform=ax.transAxes, color=INK)\n'
        '\n'
        '\n'
        'def matrix_panel(fig, result, counts, color):\n'
        '    ax = fig.add_axes([.650, .348, .280, .292])\n'
        '    ax.set(xlim=(0, 2), ylim=(2, 0))\n'
        '    ax.axis("off")\n'
        '    for col, label in enumerate(["UC", "Colonic CD"]):\n'
        '        ax.text(col+.5, -.12, label, va="bottom", ha="center", fontsize=10.2)\n'
        '    for row, label in enumerate(["UC", "Colonic CD"]):\n'
        '        ax.text(-.085, row+.5, label, ha="right", va="center", fontsize=10.2)\n'
        '    ax.text(1, -.48, "Predicted diagnosis", ha="center", va="center", fontsize=9.4, color=MUTED)\n'
        '    fig.text(.535, .494, "Observed diagnosis", rotation=90, ha="center", va="center",\n'
        '             fontsize=9.4, color=MUTED)\n'
        '    outcomes = (("true positive", "false negative"), ("false positive", "true negative"))\n'
        '    rgb = np.array(to_rgb(color))\n'
        '    for (row, col), count in np.ndenumerate(counts):\n'
        '        # A continuous light-to-dark scale is identical for both count matrices.\n'
        '        shade = (count / 24)**.65\n'
        '        face = (1-shade)*np.asarray([.95, .96, .96]) + shade*(rgb*.70)\n'
        '        ax.add_patch(Rectangle((col, row), 1, 1, ec="white", fc=face, lw=1.5))\n'
        '        linear = np.where(face <= .04045, face/12.92, ((face+.055)/1.055)**2.4)\n'
        '        luminance = linear @ np.array([.2126, .7152, .0722])\n'
        '        text_color = "white" if luminance <= .183 else "#101C23"\n'
        '        ax.text(col+.5, row+.43, str(count), ha="center", va="center", fontsize=21,\n'
        '                weight="bold", color=text_color)\n'
        '        ax.text(col+.5, row+.77, outcomes[row][col], ha="center", va="center",\n'
        '                fontsize=8, color=text_color)\n'
        '    for x, metric, label in [(.709, result.sensitivity, "Sensitivity"),\n'
        '                             (.874, result.specificity, "Specificity")]:\n'
        '        fig.text(x, .271, f"{metric:.1%}", ha="center", fontsize=15, weight="bold")\n'
        '        fig.text(x, .231, label, ha="center", fontsize=9.5, color=MUTED)\n'
        '    return ax\n'
        '\n'
        '\n'
        'def render():\n'
        '    tables = ROOT / "tables"\n'
        '    samples = pd.read_csv(tables / "selected_samples_expression_predictions.csv", float_precision="round_trip")\n'
        '    cohort = samples.loc[samples.group.isin(["UC", "CDc"])]\n'
        '    assert len(cohort) == cohort.patient_id.nunique() == 43 and sum(cohort.group == "UC") == 24\n'
        '    np.testing.assert_array_equal(cohort.UC_positive, (cohort.group == "UC").astype(int))\n'
        '    y = cohort.UC_positive.to_numpy(int)\n'
        '    for fold, (_, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(y, y), 1):\n'
        '        assert (cohort.CV_fold.iloc[test] == fold).all()\n'
        '    results = pd.read_csv(tables / "model_performance.csv").set_index("model")\n'
        '    DEST.mkdir(parents=True, exist_ok=True)\n'
        '    for name, title, color, basename in [\n'
        '        ("TYROBP_CV", "TYROBP single-gene model", "#087E78", "02_TYROBP_model_performance"),\n'
        '        ("Four_gene_CV", "Four-gene model", "#B95324", "03_four_gene_model_performance"),\n'
        '    ]:\n'
        '        result = results.loc[name]\n'
        '        counts, fpr, tpr = check_results(cohort, result)\n'
        '        fig = plt.figure(figsize=(8.8, 5.15))\n'
        '        fig.text(.08, .959, title, va="top", fontsize=21, fontfamily="Georgia")\n'
        '        fig.text(.081, .890, "GSE16879   ·   Ulcerative colitis (n = 24) vs. colonic Crohn’s disease (n = 19)",\n'
        '                 fontsize=10, color=MUTED, va="top")\n'
        '        fig.text(.13, .806, "Receiver operating characteristic", va="bottom", weight="bold", fontsize=11.2)\n'
        '        fig.text(.556, .806, "Classification at threshold 0.50", va="bottom", weight="bold", fontsize=11.2)\n'
        '        roc_panel(fig, result, fpr, tpr, color)\n'
        '        matrix_panel(fig, result, counts, color)\n'
        '        fig.add_artist(plt.Line2D([.08, .932], [.108, .108], color=GRID, lw=.8))\n'
        '        method = ("The model uses TYROBP." if name == "TYROBP_CV" else\n'
        '                  "The model uses CASP1, MMP9, PECAM1 and TYROBP.")\n'
        '        fig.text(.08, .071, method + "  The diamond marks the UC probability cutoff of 0.50.",\n'
        '                 va="center", fontsize=9, color=MUTED)\n'
        '        fig.text(.08, .032, "AUC uses pooled out-of-fold predictions from five-fold CV. The 95% CI uses 5,000 stratified resamples of these predictions.",\n'
        '                 va="center", fontsize=9, color=MUTED)\n'
        '        fig.savefig(DEST / (basename + ".pdf"))\n'
        '        fig.savefig(DEST / (basename + ".png"), dpi=450)\n'
        '        plt.close(fig)\n'
        '        print(name, counts.tolist(), result.auc, result.ci_low, result.ci_high)\n'
        '\n'
        '\n'
        'if __name__ == "__main__":\n'
        '    render()\n'
    ),
    'fig06_ab_expression_roc.py': (
        'from pathlib import Path\n'
        'import re\n'
        '\n'
        'import matplotlib\n'
        'from matplotlib import font_manager\n'
        'from matplotlib.lines import Line2D\n'
        'from matplotlib.ticker import FormatStrFormatter\n'
        'import numpy as np\n'
        'import pandas as pd\n'
        'from scipy.stats import false_discovery_control, mannwhitneyu\n'
        '\n'
        'from reference_violins import draw_group, format_panel, GENES, BLUE, RED, INK, DATA, BASE, plt\n'
        'from rev_model_figures import check_results\n'
        '\n'
        'FONT = "Times New Roman"\n'
        'for file in ("times.ttf", "timesi.ttf"):\n'
        '    font_manager.fontManager.addfont(str(Path("C:/Windows/Fonts")/file))\n'
        'plt.rcParams.update({\n'
        '    "font.family": FONT, "font.weight": "normal", "font.size": 9,\n'
        '    "axes.titleweight": "normal", "axes.labelweight": "normal",\n'
        '    "axes.linewidth": .75, "text.color": INK, "axes.edgecolor": INK,\n'
        '    "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": FONT+":italic",\n'
        '    "mathtext.bf": FONT, "mathtext.fallback": None, "mathtext.cal": FONT,\n'
        '    "mathtext.sf": FONT, "mathtext.tt": FONT, "pdf.fonttype": 42, "ps.fonttype": 42,\n'
        '})\n'
        'samples=pd.read_csv(DATA/"selected_samples_expression_predictions.csv",float_precision="round_trip")\n'
        'cohort=samples[samples.group.isin(["CDc","UC"])]\n'
        'assert len(cohort)==cohort.patient_id.nunique()==43 and cohort.UC_positive.sum()==24\n'
        'stats=pd.read_csv(DATA/"model_performance.csv").set_index("model")\n'
        'saved=pd.read_csv(DATA/"four_gene_expression_UC_vs_CDc.csv").set_index("gene")\n'
        '\n'
        'w,h=8.05,4.36\n'
        'fig=plt.figure(figsize=(w,h))\n'
        '\n'
        '\n'
        'def words(x,y,string,**kwargs):\n'
        '    return fig.text(x/w,y/h,string,**kwargs)\n'
        '\n'
        '\n'
        'def axes(x,y,width,height):\n'
        '    return fig.add_axes((x/w,y/h,width/w,height/h))\n'
        '\n'
        '\n'
        'words(.075,4.24,"A",fontsize=15,va="top")\n'
        'words(.36,4.22,"Gene expression in CD and UC",fontsize=11.6,va="top")\n'
        'words(4.50,4.24,"B",fontsize=15,va="top")\n'
        'words(4.80,4.22,"ROC curves for UC versus CD",fontsize=11.6,va="top")\n'
        '\n'
        'pairs=[tuple(samples.loc[samples.group==k,g].to_numpy() for k in ("CDc","UC")) for g in GENES]\n'
        'raw=[mannwhitneyu(a,b,alternative="two-sided").pvalue for a,b in pairs]\n'
        'adj=false_discovery_control(raw)\n'
        'np.testing.assert_allclose(adj,saved.loc[list(GENES),"BH_adjusted_p"])\n'
        'for i,(g,pair,p) in enumerate(zip(GENES,pairs,adj)):\n'
        '    ax=axes([.55,2.62][i%2],[2.40,.43][i//2],1.48,1.27)\n'
        '    format_panel(ax,g,"CDc",pair,p,r"$P$")\n'
        '    ax.set_title(g,fontsize=10.8,fontstyle="italic",pad=3)\n'
        '    ax.set_ylabel(r"Gene expression (log$_2$ scale)",fontsize=8.6,labelpad=2.5)\n'
        '    ax.tick_params(axis="x",labelsize=8.6,pad=2)\n'
        '    ax.tick_params(axis="y",labelsize=8.25,pad=2,length=2.6,width=.7)\n'
        '    for label in ax.texts:\n'
        '        label.set_fontsize(8.5)\n'
        '    groups=[part[g].to_numpy() for _,part in samples.groupby("group")]\n'
        '    centered=np.concatenate([v-v.mean() for v in groups])\n'
        '    bandwidth=np.sqrt((centered**2).sum()/(len(centered)-len(groups)))*np.median([len(v) for v in groups])**(-.2)\n'
        '    for j,(v,color) in enumerate(zip(pair,[BLUE,RED])):\n'
        '        draw_group(ax,j,v,color,bandwidth,136+11*i+j,mark_scale=.58)\n'
        '\n'
        'words(6.37,3.94,"GSE16879  ·  Five-fold cross-validation  ·  43 patients",\n'
        '      ha="center",va="center",fontsize=8.5)\n'
        'ax=axes(5.09,1.04,2.67,2.67)\n'
        'ax.set(xlim=(-.012,1.012),ylim=(-.012,1.012),aspect="equal")\n'
        'ax.set_xlabel("1 − specificity",fontsize=9.4,labelpad=3)\n'
        'ax.set_ylabel("Sensitivity",fontsize=9.4,labelpad=3)\n'
        'for side in ("left","bottom"):\n'
        '    ax.spines[side].set_bounds(0,1)\n'
        'for axis in (ax.xaxis,ax.yaxis):\n'
        '    axis.set_ticks(np.linspace(0,1,6))\n'
        '    axis.set_major_formatter(FormatStrFormatter("%.1f"))\n'
        'ax.tick_params(labelsize=8.5,length=2.6,width=.7,pad=2.5)\n'
        'ax.plot([0,1],[0,1],color="#9EA8AC",lw=.8,ls=(0,(3,3)),zorder=1)\n'
        '\n'
        'words(7.85,.568,"AUC (95% CI)",ha="right",fontsize=8.5,va="center")\n'
        'for key,name,color,ls,y in [\n'
        '    ("TYROBP_CV","TYROBP single-gene model",BLUE,"solid",.376),\n'
        '    ("Four_gene_CV","Four-gene model",RED,(0,(4,2)),.167),\n'
        ']:\n'
        '    est=stats.loc[key]\n'
        '    _,fpr,tpr=check_results(cohort,est)\n'
        '    np.testing.assert_allclose(np.trapezoid(tpr,fpr),est.auc)\n'
        '    ax.plot(fpr,tpr,color=color,lw=1.55,ls=ls,solid_joinstyle="miter",zorder=2)\n'
        '    fig.add_artist(Line2D([4.71/w,4.98/w],[y/h,y/h],transform=fig.transFigure,\n'
        '                          color=color,lw=1.55,ls=ls))\n'
        '    words(5.04,y,name,fontsize=8.75,va="center")\n'
        '    words(7.85,y,f"{est.auc:.3f} ({est.ci_low:.3f}–{est.ci_high:.3f})",\n'
        '          fontsize=8.75,ha="right",va="center")\n'
        '\n'
        'out=BASE/"outputs"/"TYROBP_AB组合图"\n'
        'out.mkdir(exist_ok=True)\n'
        'base=out/"Figure_AB_expression_ROC"\n'
        'fig.savefig(base.with_suffix(".png"),dpi=600)\n'
        'fig.savefig(base.with_suffix(".pdf"),metadata={\n'
        '    "Title":"Gene expression and ROC curves for UC versus colonic Crohn\'s disease",\n'
        '    "Subject":"Panel A shows BH-adjusted P values. Panel B shows pooled out-of-fold ROC curves.",\n'
        '})\n'
        'plt.close(fig)\n'
        'names=set(re.findall(rb"/BaseFont\\s*/([^\\s/<]+)",base.with_suffix(".pdf").read_bytes()))\n'
        'assert names and all(b"TimesNewRoman" in n and b"Bold" not in n for n in names),names\n'
        'print(sorted(names))\n'
    ),
    'fig06_full_geo_analysis.py': (
        '"""Prespecified GEO disease-control reanalysis. See analysis_plan.json.\n'
        'Run: python analyze.py --raw-dir source_data --out-dir analysis_results\n'
        'Requires numpy, pandas, scipy, scikit-learn, matplotlib.\n'
        '"""\n'
        'from pathlib import Path\n'
        'import sys\n'
        "DEPS=Path(__file__).resolve().parent.parent/'analysis_deps'\n"
        'if DEPS.exists():sys.path.insert(0,str(DEPS))\n'
        'import argparse,csv,gzip,io,json,re,platform\n'
        'from collections import Counter\n'
        'import numpy as np\n'
        'import pandas as pd\n'
        'import scipy\n'
        'from scipy.stats import mannwhitneyu\n'
        'import sklearn\n'
        'from sklearn.pipeline import make_pipeline\n'
        'from sklearn.preprocessing import StandardScaler\n'
        'from sklearn.linear_model import LogisticRegression\n'
        'from sklearn.model_selection import StratifiedKFold\n'
        'from sklearn.metrics import roc_auc_score,roc_curve,confusion_matrix,balanced_accuracy_score\n'
        'import matplotlib\n'
        "matplotlib.use('Agg')\n"
        'import matplotlib.pyplot as plt\n'
        '\n'
        "GENES=['CASP1','MMP9','PECAM1','TYROBP']\n"
        'HERE=Path(__file__).resolve().parent\n'
        'parser=argparse.ArgumentParser()\n'
        "parser.add_argument('--raw-dir',type=Path,default=HERE/'raw')\n"
        "parser.add_argument('--out-dir',type=Path,default=HERE/'results')\n"
        'args=parser.parse_args()\n'
        'RAW,OUT=args.raw_dir,args.out_dir\n'
        "TABLES,FIGS=OUT/'tables',OUT/'figures'\n"
        'for p in [OUT,TABLES,FIGS]:p.mkdir(exist_ok=True,parents=True)\n'
        '\n'
        'def save_json(obj,path):\n'
        "    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')\n"
        '\n'
        'def read_matrix(name):\n'
        '    # Full decompression validates the gzip trailer/CRC before any statistics.\n'
        "    text=gzip.decompress((RAW/name).read_bytes()).decode('utf-8')\n"
        "    before,body=text.split('!series_matrix_table_begin',1)\n"
        "    table=body.split('!series_matrix_table_end',1)[0].strip()\n"
        "    data=pd.read_csv(io.StringIO(table),sep='\\t',index_col=0)\n"
        '    data.index=data.index.astype(str)\n'
        "    rows=[next(csv.reader([l],delimiter='\\t')) for l in before.splitlines() if l.startswith('!Sample_')]\n"
        "    acc=next(r[1:] for r in rows if r[0]=='!Sample_geo_accession')\n"
        '    meta=pd.DataFrame(index=acc)\n'
        '    for r in rows:\n'
        "        key=r[0].removeprefix('!Sample_')\n"
        "        if key=='characteristics_ch1':\n"
        "            keys={x.split(':',1)[0].strip() for x in r[1:]}\n"
        '            assert len(keys)==1,keys\n'
        '            key=keys.pop()\n'
        "            values=[x.split(':',1)[1].strip() if ':' in x else x for x in r[1:]]\n"
        '        else:values=r[1:]\n'
        '        if len(values)==len(meta):meta[key]=values\n'
        '    assert list(data.columns)==list(meta.index)\n'
        '    return data,meta\n'
        '\n'
        'def annotate():\n'
        "    text=gzip.decompress((RAW/'GPL570.annot.gz').read_bytes()).decode('utf-8')\n"
        "    tab=text.split('!platform_table_begin',1)[1].split('!platform_table_end',1)[0].strip()\n"
        "    ann=pd.read_csv(io.StringIO(tab),sep='\\t',dtype=str,keep_default_na=False,usecols=['ID','Gene symbol','Gene ID','Gene title'])\n"
        "    ann['symbols']=ann['Gene symbol'].apply(lambda s:sorted(set(re.split(r'\\s*///\\s*',s))))\n"
        '    records=[]\n'
        '    for _,r in ann.iterrows():\n'
        "        intersect=set(r['symbols'])&set(GENES)\n"
        '        if intersect:\n'
        "            for g in sorted(intersect):records.append({'gene':g,'probe':r['ID'],'annotation_symbols':r['Gene symbol'],'entrez_ids':r['Gene ID'],'included_unambiguous':r['symbols']==[g]})\n"
        '    return pd.DataFrame(records)\n'
        '\n'
        'mapping=annotate()\n'
        "target,tm=read_matrix('GSE16879_series_matrix.txt.gz')\n"
        "source,sm=read_matrix('GSE38713_series_matrix.txt.gz')\n"
        "print('Target',target.shape,'Source',source.shape,flush=True)\n"
        'original_target_probes=target.shape[0]\n'
        'quantiles={}\n'
        "for name,dat in [('GSE16879',target),('GSE38713',source)]:\n"
        '    q=np.nanquantile(dat.to_numpy(),[0,.01,.5,.99,1])\n'
        "    quantiles[name]=dict(zip(['min','q01','median','q99','max'],map(float,q)))\n"
        "probe_sets={g:mapping.loc[(mapping.gene==g)&mapping.included_unambiguous,'probe'].tolist() for g in GENES}\n"
        'for g,probes in probe_sets.items():\n'
        '    assert probes and set(probes)<=set(target.index) and set(probes)<=set(source.index),(g,probes)\n'
        "mapping['in_target_matrix']=mapping.probe.isin(target.index)\n"
        "mapping['in_source_matrix']=mapping.probe.isin(source.index)\n"
        "mapping.to_csv(TABLES/'probe_mapping.csv',index=False,encoding='utf-8-sig')\n"
        'all_probes=sorted(set(sum(probe_sets.values(),[])))\n'
        '# The full target matrix has 16 zero values in unrelated probes. All 12\n'
        '# prespecified feature probes are positive. Transform only these probes, with\n'
        '# no pseudocount, imputation or outcome-dependent filtering.\n'
        'assert target.loc[all_probes].to_numpy().min()>0 and target.to_numpy().max()>100\n'
        'assert source.to_numpy().max()<100\n'
        'target=np.log2(target.loc[all_probes])\n'
        'source=source.loc[all_probes]\n'
        '# Source matrix explicitly states GC-RMA; it is already on a log2 scale.\n'
        '\n'
        "def collapse(dat,how='median'):\n"
        '    return pd.DataFrame({g:getattr(dat.loc[probes],how)(axis=0) for g,probes in probe_sets.items()})\n'
        '\n'
        'def target_group(title):\n'
        "    if re.fullmatch(r'UC(?:NR|R)\\d+_beforeT',title):return 'UC'\n"
        "    if re.fullmatch(r'CDc(?:NR|R)\\d+_beforeT',title):return 'CDc'\n"
        "    if re.fullmatch(r'CO[1-6]',title):return 'CO'\n"
        "    return 'Excluded'\n"
        '\n'
        "tm['group']=tm.title.map(target_group)\n"
        "tm['patient_id']=tm.title.str.replace(r'_(before|after)T$','',regex=True)\n"
        "tm['included']=tm.group!='Excluded'\n"
        "tm['exclusion_reason']=np.where(tm.included,'',np.where(tm.title.str.contains('_afterT'),'Post-infliximab paired biopsy',np.where(tm.title.str.startswith('CDi'),'Ileal Crohn disease','Ileal control / other non-colonic-baseline sample')))\n"
        "tm.index.name='GSM'\n"
        "tm[['title','patient_id','tissue','disease','response to infliximab','before or after first infliximab treatment','group','included','exclusion_reason']].to_csv(TABLES/'all_samples_inclusion_audit.csv',encoding='utf-8-sig')\n"
        'included=tm[tm.included].copy()\n'
        "assert Counter(included.group)=={'UC':24,'CDc':19,'CO':6}\n"
        "assert included['tissue'].eq('Colon').all() and included.patient_id.is_unique\n"
        "expected_file=HERE/'expected_samples_GSE16879.json'\n"
        'if expected_file.exists():\n'
        "    expected=json.loads(expected_file.read_text(encoding='utf-8'))\n"
        "    assert set(included.index)==set(expected['included_gsm'])\n"
        '\n'
        'tx=collapse(target).loc[included.index]\n'
        'sx=collapse(source)\n'
        "print('Source groups',sm.source_name_ch1.value_counts().to_dict(),flush=True)\n"
        "sm['source_group']=sm.source_name_ch1\n"
        "sm['included']=~sm.source_name_ch1.str.contains('uninvolved|non-involved',case=False,regex=True)\n"
        "sm['UC']=~sm.source_name_ch1.str.contains('control|healthy',case=False,regex=True)\n"
        'train=sm[sm.included]\n'
        "assert len(train)==36 and int(train.UC.sum())==23,train[['title','source_group','UC']]\n"
        "sm[['title','source_group','included','UC']].to_csv(TABLES/'source_GSE38713_selection.csv',encoding='utf-8-sig')\n"
        '\n'
        'def model():\n'
        "    return make_pipeline(StandardScaler(),LogisticRegression(C=1,penalty='l2',solver='lbfgs',max_iter=2000,random_state=42))\n"
        '\n'
        "main=included[included.group.isin(['UC','CDc'])].copy()\n"
        'X=tx.loc[main.index,GENES].to_numpy()\n'
        "y=main.group.eq('UC').astype(int).to_numpy()\n"
        '\n'
        'def cv_predict(X,y,seed):\n'
        '    splits=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=seed).split(X,y))\n'
        "    scores={name:np.zeros(len(y)) for name in ['TYROBP_CV','Four_gene_CV']}\n"
        '    probs={name:np.zeros(len(y)) for name in scores}\n'
        '    folds=np.zeros(len(y),int)\n'
        '    fold_models=[]\n'
        '    for fold,(tr,te) in enumerate(splits,1):\n'
        '        assert not set(tr)&set(te)\n'
        '        folds[te]=fold\n'
        "        for name,cols in [('TYROBP_CV',[3]),('Four_gene_CV',[0,1,2,3])]:\n"
        '            pipe=model().fit(X[tr][:,cols],y[tr])\n'
        '            scores[name][te]=pipe.decision_function(X[te][:,cols])\n'
        '            probs[name][te]=pipe.predict_proba(X[te][:,cols])[:,1]\n'
        "            fold_models.append({'fold':fold,'model':name,'train_GSM':main.index[tr].tolist(),'test_GSM':main.index[te].tolist(),'train_n':len(tr),'test_n':len(te),'coef':pipe[-1].coef_[0].tolist(),'intercept':float(pipe[-1].intercept_[0]),'mean':pipe[0].mean_.tolist(),'scale':pipe[0].scale_.tolist()})\n"
        '    return scores,probs,folds,fold_models\n'
        '\n'
        'scores,probs,folds,fold_models=cv_predict(X,y,42)\n'
        "scores={'TYROBP_expression':X[:,3],**scores}\n"
        "save_json(fold_models,OUT/'fold_models.json')\n"
        '\n'
        'def pair_auc(pos,neg):\n'
        '    return float(((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:])).mean())\n'
        '\n'
        'def bootstrap(y,scores,n=5000,seed=20261118):\n'
        '    rng=np.random.default_rng(seed)\n'
        '    pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)\n'
        '    idx=np.concatenate([rng.choice(pos,(n,len(pos)),replace=True),rng.choice(neg,(n,len(neg)),replace=True)],axis=1)\n'
        '    result={}\n'
        '    arrays={}\n'
        '    for name,s in scores.items():\n'
        '        p=np.asarray(s)[idx[:,:len(pos)]]\n'
        '        q=np.asarray(s)[idx[:,len(pos):]]\n'
        '        vals=((p[:,:,None]>q[:,None,:])+.5*(p[:,:,None]==q[:,None,:])).mean(axis=(1,2))\n'
        '        arrays[name]=vals\n'
        "        result[name]={'auc':float(roc_auc_score(y,s)),'ci_low':float(np.quantile(vals,.025)),'ci_high':float(np.quantile(vals,.975))}\n"
        "        assert abs(result[name]['auc']-pair_auc(np.asarray(s)[pos],np.asarray(s)[neg]))<1e-12\n"
        '    return result,arrays\n'
        '\n'
        'primary,boots=bootstrap(y,scores)\n'
        'rows=[]\n'
        'for name,r in primary.items():\n'
        "    row={'analysis':'Primary UC versus CDc','model':name,'n_positive':24,'n_negative':19,**r,'validation':'Direct prespecified marker' if name=='TYROBP_expression' else 'New UC-vs-CDc model, internal 5-fold OOF','score_direction':'Higher expression predicts UC' if name=='TYROBP_expression' else 'Training-fold fitted UC score','ci_method':'5000 stratified bootstrap of fixed scores; OOF CI conditional on split' if name!='TYROBP_expression' else '5000 stratified patient bootstrap'}\n"
        '    if name in probs:\n'
        '        pred=probs[name]>=.5\n'
        '        tn,fp,fn,tp=confusion_matrix(y,pred).ravel()\n'
        "        row.update({'threshold':.5,'TP':int(tp),'FN':int(fn),'TN':int(tn),'FP':int(fp),'sensitivity':float(tp/(tp+fn)),'specificity':float(tn/(tn+fp)),'balanced_accuracy':float(balanced_accuracy_score(y,pred))})\n"
        '    rows.append(row)\n'
        "delta=boots['Four_gene_CV']-boots['TYROBP_CV']\n"
        "delta_result={'comparison':'Four_gene_CV minus TYROBP_CV','delta_auc':primary['Four_gene_CV']['auc']-primary['TYROBP_CV']['auc'],'ci_low':float(np.quantile(delta,.025)),'ci_high':float(np.quantile(delta,.975)),'interpretation':'Paired bootstrap conditional on the same OOF predictions; no proof of added clinical value'}\n"
        '\n'
        '# Repeated splits assess instability without choosing the best split.\n'
        'repeats=[]\n'
        'for seed in range(42,142):\n'
        '    sc,_,_,_=cv_predict(X,y,seed)\n'
        "    repeats.append({'seed':seed,**{name:float(roc_auc_score(y,s)) for name,s in sc.items()}})\n"
        'repeated=pd.DataFrame(repeats)\n'
        "repeated.to_csv(TABLES/'repeated_5fold_CV_100_seeds.csv',index=False,encoding='utf-8-sig')\n"
        "stability={name:{'median':float(repeated[name].median()),'q025':float(repeated[name].quantile(.025)),'q975':float(repeated[name].quantile(.975)),'min':float(repeated[name].min()),'max':float(repeated[name].max())} for name in ['TYROBP_CV','Four_gene_CV']}\n"
        '\n'
        '# Reconstructed source models are sensitivity analyses, never asserted identical\n'
        '# to the original manuscript nomogram. No target fitting or target normalization.\n'
        'source_models=[]\n'
        'transport_scores={}\n'
        "for name,cols in [('TYROBP_reconstructed_source',['TYROBP']),('Four_gene_reconstructed_source',GENES)]:\n"
        '    pipe=model().fit(sx.loc[train.index,cols].to_numpy(),train.UC.astype(int).to_numpy())\n'
        '    transport_scores[name]=pipe.decision_function(tx.loc[main.index,cols].to_numpy())\n'
        "    source_models.append({'model':name,'features':cols,'source':'GSE38713 23UC/13nonIBD','mean':pipe[0].mean_.tolist(),'scale':pipe[0].scale_.tolist(),'coef':pipe[-1].coef_[0].tolist(),'intercept':float(pipe[-1].intercept_[0]),'warning':'Reconstructed sensitivity model, not verified original nomogram; source GC-RMA and target GCOS-log2 differ'})\n"
        'frozen,frozenboots=bootstrap(y,transport_scores)\n'
        'for name,r in frozen.items():\n'
        "    rows.append({'analysis':'Frozen reconstructed source-model sensitivity, UC versus CDc','model':name,'n_positive':24,'n_negative':19,**r,'validation':'Source-fitted then frozen; NOT verified original nomogram','score_direction':'Frozen source UC-vs-nonIBD score','ci_method':'5000 target-patient stratified bootstrap of fixed scores'})\n"
        "save_json(source_models,OUT/'reconstructed_source_models.json')\n"
        '\n'
        '# Original probe-level values and mean-collapse sensitivity are supplied.\n'
        'all_probes=sorted(set(sum(probe_sets.values(),[])))\n'
        "target.loc[all_probes,included.index].T.rename_axis('GSM').to_csv(TABLES/'log2_probe_expression.csv',encoding='utf-8-sig')\n"
        "mean_tx=collapse(target,'mean').loc[main.index]\n"
        'mean_scores,_,_,_=cv_predict(mean_tx[GENES].to_numpy(),y,42)\n'
        "probe_sensitivity={'aggregation':'Mean log2 expression across the same unambiguous probes; no target-outcome selection','TYROBP_expression_auc':float(roc_auc_score(y,mean_tx.TYROBP)),**{name+'_auc':float(roc_auc_score(y,s)) for name,s in mean_scores.items()}}\n"
        '\n'
        '# Descriptive healthy comparisons are explicitly secondary.\n'
        'secondary=[]\n'
        "for group in ['UC','CDc']:\n"
        "    sub=included[included.group.isin([group,'CO'])]\n"
        '    sy=sub.group.eq(group).astype(int).to_numpy()\n'
        "    vals=tx.loc[sub.index,'TYROBP'].to_numpy()\n"
        "    res,_=bootstrap(sy,{'TYROBP_expression':vals},seed=20261119)\n"
        "    stat=mannwhitneyu(vals[sy==1],vals[sy==0],alternative='two-sided',method='auto')\n"
        "    secondary.append({'comparison':group+' versus CO','n_positive':int(sy.sum()),'n_negative':int((sy==0).sum()),**res['TYROBP_expression'],'mannwhitney_p':float(stat.pvalue),'note':'Secondary descriptive comparison; six controls only'})\n"
        '\n'
        'expr=[]\n'
        'for g in GENES:\n'
        "    u=tx.loc[included.group=='UC',g].to_numpy()\n"
        "    c=tx.loc[included.group=='CDc',g].to_numpy()\n"
        "    stat=mannwhitneyu(u,c,alternative='two-sided',method='auto')\n"
        "    row={'gene':g,'UC_n':len(u),'CDc_n':len(c),'UC_median':float(np.median(u)),'UC_q25':float(np.quantile(u,.25)),'UC_q75':float(np.quantile(u,.75)),'CDc_median':float(np.median(c)),'CDc_q25':float(np.quantile(c,.25)),'CDc_q75':float(np.quantile(c,.75)),'median_log2_difference_UC_minus_CDc':float(np.median(u)-np.median(c)),'mannwhitney_p':float(stat.pvalue)}\n"
        '    expr.append(row)\n'
        "pv=np.array([r['mannwhitney_p'] for r in expr]);order=np.argsort(pv)\n"
        'adj=np.minimum.accumulate((pv[order]*4/np.arange(1,5))[::-1])[::-1]\n'
        "for i,j in enumerate(order):expr[j]['BH_adjusted_p']=float(min(adj[i],1))\n"
        '\n'
        "sample_out=included[['title','patient_id','group','response to infliximab']].join(tx)\n"
        "sample_out['UC_positive']=sample_out.group.map({'UC':1,'CDc':0})\n"
        "sample_out.loc[main.index,'CV_fold']=folds\n"
        "for name,s in scores.items():sample_out.loc[main.index,name+'_score']=s\n"
        "for name,p in probs.items():sample_out.loc[main.index,name+'_probability']=p\n"
        "for name,s in transport_scores.items():sample_out.loc[main.index,name+'_score']=s\n"
        "sample_out.to_csv(TABLES/'selected_samples_expression_predictions.csv',encoding='utf-8-sig')\n"
        "pd.DataFrame(rows).to_csv(TABLES/'model_performance.csv',index=False,encoding='utf-8-sig')\n"
        "pd.DataFrame(expr).to_csv(TABLES/'four_gene_expression_UC_vs_CDc.csv',index=False,encoding='utf-8-sig')\n"
        "pd.DataFrame(secondary).to_csv(TABLES/'secondary_TYROBP_healthy_comparisons.csv',index=False,encoding='utf-8-sig')\n"
        "pd.DataFrame([delta_result]).to_csv(TABLES/'paired_delta_auc.csv',index=False,encoding='utf-8-sig')\n"
        '\n'
        "plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})\n"
        "colors={'CO':'#8A929A','CDc':'#277DA8','UC':'#BD4F56'}\n"
        '\n'
        'def savefig(fig,name):\n'
        "    fig.savefig(FIGS/(name+'.png'),dpi=300,bbox_inches='tight')\n"
        "    fig.savefig(FIGS/(name+'.pdf'),bbox_inches='tight')\n"
        '    plt.close(fig)\n'
        '\n'
        'fig,axes=plt.subplots(2,2,figsize=(8,6.8),constrained_layout=True)\n'
        'rng=np.random.default_rng(20261118)\n'
        'for ax,g in zip(axes.flat,GENES):\n'
        "    groups=['CO','CDc','UC']\n"
        '    vals=[tx.loc[included.group==k,g].to_numpy() for k in groups]\n'
        "    bp=ax.boxplot(vals,positions=np.arange(3),widths=.5,showfliers=False,patch_artist=True,medianprops={'color':'black','linewidth':1.4})\n"
        "    for patch,k in zip(bp['boxes'],groups):patch.set_facecolor(colors[k]);patch.set_alpha(.28)\n"
        "    for x,k,v in zip(range(3),groups,vals):ax.scatter(x+rng.uniform(-.13,.13,len(v)),v,s=24,color=colors[k],alpha=.85,edgecolor='white',linewidth=.3)\n"
        "    r=next(r for r in expr if r['gene']==g)\n"
        '    ax.set_title(g+\'   UC vs CDc BH P = \'+f"{r[\'BH_adjusted_p\']:.3g}",fontsize=11)\n'
        "    ax.set_xticks(range(3),['Colon control\\n(n=6)','Colonic CD\\n(n=19)','UC\\n(n=24)'])\n"
        "    ax.set_ylabel('Log2 GCOS signal')\n"
        "fig.suptitle('GSE16879 baseline colonic biopsies',fontsize=14)\n"
        "savefig(fig,'Figure_1_four_gene_expression')\n"
        '\n'
        'fig,ax=plt.subplots(figsize=(6.5,6))\n'
        "labels={'TYROBP_expression':'TYROBP expression, high = UC','TYROBP_CV':'TYROBP model, 5-fold OOF','Four_gene_CV':'Four-gene model, 5-fold OOF'}\n"
        "for name,col,ls in [('TYROBP_expression','#808080','--'),('TYROBP_CV','#277DA8','-'),('Four_gene_CV','#BD4F56','-')]:\n"
        '    fpr,tpr,_=roc_curve(y,scores[name]);r=primary[name]\n'
        '    ax.plot(fpr,tpr,color=col,linestyle=ls,lw=2,label=labels[name]+f"\\nAUC {r[\'auc\']:.3f} (95% CI {r[\'ci_low\']:.3f}–{r[\'ci_high\']:.3f})")\n'
        "ax.plot([0,1],[0,1],color='#C0C0C0',lw=1)\n"
        "ax.set(xlabel='1 − specificity',ylabel='Sensitivity',xlim=(0,1),ylim=(0,1.02),title='UC versus colonic Crohn disease\\n24 UC and 19 CDc, GSE16879')\n"
        "ax.legend(loc='lower right',fontsize=8.5,frameon=False)\n"
        "fig.text(.14,.005,'OOF intervals are conditional on the selected held-out predictions.',fontsize=8)\n"
        "savefig(fig,'Figure_2_primary_UC_vs_CDc_ROC')\n"
        '\n'
        'fig,ax=plt.subplots(figsize=(6.5,6))\n'
        "for name,col,label in [('TYROBP_reconstructed_source','#277DA8','TYROBP source model'),('Four_gene_reconstructed_source','#BD4F56','Four-gene source model')]:\n"
        '    fpr,tpr,_=roc_curve(y,transport_scores[name]);r=frozen[name]\n'
        '    ax.plot(fpr,tpr,color=col,lw=2,label=label+f"\\nAUC {r[\'auc\']:.3f} (95% CI {r[\'ci_low\']:.3f}–{r[\'ci_high\']:.3f})")\n'
        "ax.plot([0,1],[0,1],color='#C0C0C0',lw=1)\n"
        "ax.set(xlabel='1 − specificity',ylabel='Sensitivity',xlim=(0,1),ylim=(0,1.02),title='Sensitivity analysis of frozen reconstructed models\\nSource GSE38713; target GSE16879 UC versus CDc')\n"
        "ax.legend(loc='lower right',fontsize=9,frameon=False)\n"
        "fig.text(.10,.005,'Reconstructed models are not verified copies of the original nomogram.',fontsize=8)\n"
        "savefig(fig,'Figure_3_reconstructed_model_sensitivity')\n"
        '\n'
        'fig,ax=plt.subplots(figsize=(6.2,4.6))\n'
        "for x,name,col in [(0,'TYROBP_CV','#277DA8'),(1,'Four_gene_CV','#BD4F56')]:\n"
        '    v=repeated[name].to_numpy()\n'
        '    ax.scatter(x+rng.uniform(-.11,.11,len(v)),v,alpha=.22,color=col,s=14)\n'
        'bp=ax.boxplot([repeated.TYROBP_CV,repeated.Four_gene_CV],positions=[0,1],widths=.42,showfliers=False,patch_artist=True)\n'
        "for p,col in zip(bp['boxes'],['#277DA8','#BD4F56']):p.set(facecolor=col,alpha=.2)\n"
        "ax.axhline(.5,ls='--',color='#999999',lw=1)\n"
        "ax.set(xticks=[0,1],xticklabels=['TYROBP model','Four-gene model'],ylabel='Pooled out-of-fold AUC',ylim=(0,1),title='Sensitivity to 100 prespecified five-fold splits')\n"
        "fig.text(.10,.01,'Each point is one split seed. The distribution is not a confidence interval.',fontsize=8)\n"
        "savefig(fig,'Figure_4_cross_validation_stability')\n"
        '\n'
        "summary={'target':'GSE16879','counts':dict(Counter(included.group)),'target_total_matrix_samples':target.shape[1],'target_probes':target.shape[0],'source_matrix_samples':source.shape[1],'source_selected_n':36,'source_UC_n':23,'source_nonIBD_n':13,'probe_sets':probe_sets,'data_scale_quantiles_before_transform':quantiles,'primary':primary,'model_performance':rows,'paired_delta_auc':delta_result,'repeated_cv_stability':stability,'source_model_sensitivity':frozen,'probe_aggregation_sensitivity':probe_sensitivity,'gene_expression':expr,'secondary_healthy_comparisons':secondary,'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,'matplotlib':matplotlib.__version__},'sources':{'GSE16879':'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE16879','GSE16879_original_article':'https://doi.org/10.1371/journal.pone.0007984','GSE38713':'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE38713','GPL570':'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL570'},'samples':sample_out.reset_index().replace({np.nan:None}).to_dict(orient='records'),'probe_mapping':mapping.to_dict(orient='records')}\n"
        "summary['target_probes']=original_target_probes\n"
        "save_json(summary,OUT/'analysis_summary.json')\n"
        "print(json.dumps({k:summary[k] for k in ['counts','probe_sets','primary','paired_delta_auc','source_model_sensitivity','repeated_cv_stability','secondary_healthy_comparisons']},ensure_ascii=False,indent=2),flush=True)\n"
    ),
    'fig09_beautify_figures.py': (
        '"""Redraw clinical expression figures to match the supplied publication style.\n'
        '\n'
        'Run: python beautify_figures.py\n'
        'Only display files and the report image links are changed. Expression and test\n'
        'results are read from the existing audited tables and are not recalculated.\n'
        '"""\n'
        'from pathlib import Path\n'
        'import json\n'
        'import hashlib\n'
        'import numpy as np\n'
        'import pandas as pd\n'
        'from scipy.stats import gaussian_kde\n'
        'import matplotlib\n'
        "matplotlib.use('Agg')\n"
        'import matplotlib.pyplot as plt\n'
        'from matplotlib.ticker import MaxNLocator, FormatStrFormatter\n'
        '\n'
        'ROOT = Path(__file__).resolve().parent\n'
        "OUT = ROOT/'figures'/'violin_style'\n"
        "NAVY, WINE, PALE = '#173F63', '#B01D42', '#8DA9BF'\n"
        "INK = '#202C35'\n"
        'SEED = 20261115\n'
        '\n'
        'def p_label(value, adjusted=True):\n'
        '    # Display only P, as requested; the correction method remains in the caption.\n'
        "    name = 'P'\n"
        '    if value < .001:\n'
        "        return rf'${name} < 0.001$'\n"
        "    return rf'${name} = {value:.2f}$'\n"
        '\n'
        'def theme():\n'
        '    plt.rcParams.update({\n'
        "        'font.family':'DejaVu Sans','font.size':10.5,\n"
        "        'axes.labelsize':11,'axes.titlesize':10.5,'axes.labelcolor':INK,\n"
        "        'text.color':INK,'xtick.color':INK,'ytick.color':INK,\n"
        "        'axes.edgecolor':INK,'axes.linewidth':1.0,\n"
        "        'xtick.major.size':3.3,'ytick.major.size':3.3,\n"
        "        'xtick.major.width':.85,'ytick.major.width':.85,\n"
        "        'pdf.fonttype':42,'svg.fonttype':'none',\n"
        "        'savefig.facecolor':'white','figure.facecolor':'white',\n"
        "        'mathtext.fontset':'dejavusans',\n"
        '    })\n'
        '\n'
        'def bracket(ax, x1, x2, y, height, text, size=9.5):\n'
        '    ax.plot([x1,x1,x2,x2],[y-height,y,y,y-height],lw=.85,c=INK,clip_on=False)\n'
        "    ax.text((x1+x2)/2,y+height*.35,text,ha='center',va='bottom',fontsize=size)\n"
        '\n'
        'def violin(ax, arrays, names, colors, limits, comparisons, rng, title=None, letter=None):\n'
        '    arrays = [np.asarray(v,dtype=float) for v in arrays]\n'
        '    assert all(len(v)>1 and np.isfinite(v).all() for v in arrays)\n'
        '    positions = np.arange(len(arrays))\n'
        '    bodies = ax.violinplot(arrays,positions=positions,widths=.72,\n'
        '                           showmeans=False,showmedians=False,showextrema=False,\n'
        '                           points=256,bw_method=.45)\n'
        "    for body,color in zip(bodies['bodies'],colors):\n"
        '        body.set(facecolor=color,edgecolor=color,alpha=1,linewidth=.65,zorder=1)\n'
        '    for i,(v,color) in enumerate(zip(arrays,colors)):\n'
        '        # Density-aware jitter keeps observations largely within the violin.\n'
        '        density=gaussian_kde(v,bw_method=.45)(v)\n'
        '        spread=.16*density/density.max()\n'
        '        jitter=rng.uniform(-1,1,len(v))*spread\n'
        '        # Keep every dot visible beside the central white box for small groups.\n'
        '        if len(v)<=10:\n'
        '            jitter=np.where(np.abs(jitter)<.082,np.where(jitter>=0,.082,-.082),jitter)\n'
        '        x=i+jitter\n'
        '        ax.scatter(x,v,s=6 if len(v)>30 else 9,facecolor=color,\n'
        "                   edgecolor='white',linewidth=.24,alpha=.92,zorder=3)\n"
        '    boxes=ax.boxplot(arrays,positions=positions,widths=.115,patch_artist=True,\n'
        '                     whis=1.5,showfliers=False,showcaps=False,\n'
        "                     boxprops={'facecolor':'white','edgecolor':INK,'linewidth':.85,'zorder':5},\n"
        "                     medianprops={'color':INK,'linewidth':1.7,'zorder':6},\n"
        "                     whiskerprops={'color':INK,'linewidth':.8,'zorder':4})\n"
        "    ax.set_xticks(positions,[f'{label}\\n(n = {len(v)})' for label,v in zip(names,arrays)])\n"
        "    ax.set_ylabel('TYROBP expression (log$_2$)')\n"
        '    ax.set_xlim(-.57,len(arrays)-.43)\n'
        '    ax.set_ylim(*limits)\n'
        '    ax.yaxis.set_major_locator(MaxNLocator(nbins=5,min_n_ticks=3,steps=[1,2,2.5,5,10]))\n'
        "    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))\n"
        "    ax.spines[['top','right']].set_visible(False)\n"
        '    ax.grid(False)\n'
        '    for x1,x2,y,label in comparisons:\n'
        '        bracket(ax,x1,x2,y,(limits[1]-limits[0])*.019,label)\n'
        '    if title:\n'
        "        ax.set_title(title,pad=9,fontweight='normal')\n"
        '    if letter:\n'
        "        ax.annotate(letter,xy=(0,1),xycoords='axes fraction',xytext=(-33,9),\n"
        "                    textcoords='offset points',ha='left',va='bottom',\n"
        "                    fontsize=15,fontweight='bold',annotation_clip=False)\n"
        '\n'
        'def save(fig,stem):\n'
        "    for ext in ('png','pdf','svg'):\n"
        "        fig.savefig(OUT/f'{stem}.{ext}',dpi=600,bbox_inches='tight',pad_inches=.06)\n"
        "    if stem=='00_TYROBP_clinical_violin_panels_A-E':\n"
        "        fig.savefig(OUT/f'{stem}_preview.png',dpi=150,bbox_inches='tight',pad_inches=.06)\n"
        '    plt.close(fig)\n'
        '\n'
        'def main():\n'
        '    OUT.mkdir(parents=True,exist_ok=True)\n'
        '    theme()\n'
        "    stats_files=[ROOT/'tables'/'activity_comparisons.csv',\n"
        "                 ROOT/'tables'/'baseline_response_comparisons.csv',\n"
        "                 ROOT/'tables'/'GSE75214_analyzed_samples.csv']\n"
        '    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in stats_files}\n'
        '    activity=pd.read_csv(stats_files[2])\n'
        '    activity_tests=pd.read_csv(stats_files[0])\n'
        '    response_tests=pd.read_csv(stats_files[1])\n'
        "    groups=['Control','Inactive UC','Active UC']\n"
        "    av=[activity.loc[activity.group.eq(g),'TYROBP_log2'].to_numpy() for g in groups]\n"
        '    assert [len(x) for x in av]==[11,23,74]\n'
        '    span=max(np.ptp(np.concatenate(av)),1.5)\n'
        '    ymax=np.max(np.concatenate(av))\n'
        '    activity_limits=(np.min(np.concatenate(av))-.10*span,ymax+.52*span)\n'
        '    acomp=[]\n'
        '    for order,(i,j,idx) in enumerate([(0,1,0),(1,2,2),(0,2,1)]):\n'
        "        p=activity_tests.iloc[idx]['p_Holm_3_activity_comparisons']\n"
        '        acomp.append((i,j,ymax+span*(.10+.145*order),p_label(p)))\n'
        '    rng=np.random.default_rng(SEED)\n'
        '    fig,ax=plt.subplots(figsize=(4.7,4.1))\n'
        '    violin(ax,av,groups,[NAVY,PALE,WINE],activity_limits,acomp,rng)\n'
        '    fig.tight_layout(pad=.6)\n'
        "    save(fig,'01_TYROBP_activity_violin')\n"
        '\n'
        '    records=[]\n'
        '    data={}\n'
        '    for _,row in response_tests.iterrows():\n'
        "        key=f'{row.dataset}_{row.endpoint}'\n"
        "        if row.dataset=='GSE16879':\n"
        "            d=pd.read_csv(ROOT/'datasets'/'gse16879'/'UC_baseline_TYROBP.csv')\n"
        "            values=[d.loc[d.response_group.eq(g),'TYROBP_log2'].to_numpy() for g in ['Responder','Nonresponder']]\n"
        '        else:\n'
        "            d=pd.read_csv(ROOT/'datasets'/'gse73661'/'results'/f'{row.endpoint}_paired_samples.csv')\n"
        "            values=[d.loc[d.response.eq(g),'TYROBP_log2_RMA_baseline'].to_numpy() for g in ['Responder','Nonresponder']]\n"
        '        assert [len(x) for x in values]==[int(row.n_R),int(row.n_NR)]\n'
        '        data[key]=(row,values)\n'
        '    # Share the expression range across all GSE73661 baseline panels.\n'
        "    all736=np.concatenate([v for key,(_,values) in data.items() if key.startswith('GSE73661') for v in values])\n"
        '    for index,(key,(row,values)) in enumerate(data.items()):\n'
        "        joint=all736 if row.dataset=='GSE73661' else np.concatenate(values)\n"
        '        spread=max(np.ptp(joint),1.5)\n'
        '        low,high=np.min(joint)-.11*spread,np.max(joint)+.26*spread\n'
        "        primary=row.family=='primary_baseline'\n"
        '        p=row.p_Holm_4_primary_baseline_comparisons if primary else row.p_raw\n'
        '        comp=[(0,1,np.max(joint)+.15*spread,p_label(p,primary))]\n'
        '        # The filenames and caption identify the dataset/drug/window; individual\n'
        "        # panels keep the same sparse typography as the user's reference.\n"
        '        fig,ax=plt.subplots(figsize=(3.7,3.6))\n'
        "        violin(ax,values,['Responder','Nonresponder'],[NAVY,WINE],(low,high),comp,rng)\n"
        '        fig.tight_layout(pad=.6)\n'
        "        stem=f'{index+3:02d}_{key}_violin'\n"
        '        save(fig,stem)\n'
        "        records.append({'file_stem':stem,'dataset':row.dataset,'endpoint':row.endpoint,\n"
        "                        'n_responder':int(row.n_R),'n_nonresponder':int(row.n_NR),\n"
        "                        'P_display_type':'Holm-adjusted across four primary baseline comparisons' if primary else 'Raw P; secondary endpoint',\n"
        "                        'P_value':float(p),'P_label':p_label(p,primary),\n"
        '                        \'short_title\':f\'{row.dataset} · {"IFX" if row.treatment=="Infliximab" else "VDZ"} \'+\n'
        "                                      {'IFX_W4_W6':'W4–6','VDZ_W6':'W6','VDZ_W12':'W12','VDZ_W52':'W52'}[row.endpoint],\n"
        "                        'limits':(low,high),'comparisons':comp,'key':key})\n"
        "    primary=[r for r in records if 'four primary' in r['P_display_type']]\n"
        '    fig,axes=plt.subplots(2,2,figsize=(7.6,7.6))\n'
        '    for ax,rec in zip(axes.ravel(),primary):\n'
        "        violin(ax,data[rec['key']][1],['Responder','Nonresponder'],[NAVY,WINE],\n"
        "               rec['limits'],rec['comparisons'],rng,title=rec['short_title'])\n"
        '    fig.subplots_adjust(hspace=.40,wspace=.38,left=.10,right=.985,bottom=.09,top=.95)\n'
        "    save(fig,'02_TYROBP_response_violin')\n"
        '\n'
        '    # A–E assembly for use beside the earlier Figure 6 panels.\n'
        '    fig=plt.figure(figsize=(7.8,10.7))\n'
        '    grid=fig.add_gridspec(3,2,height_ratios=[1,1,1],hspace=.49,wspace=.39,\n'
        '                          left=.10,right=.98,bottom=.065,top=.97)\n'
        '    a=fig.add_subplot(grid[0,:])\n'
        "    violin(a,av,groups,[NAVY,PALE,WINE],activity_limits,acomp,rng,title='GSE75214',letter='A')\n"
        '    for ax,rec,letter in zip([fig.add_subplot(grid[1,0]),fig.add_subplot(grid[1,1]),\n'
        "                             fig.add_subplot(grid[2,0]),fig.add_subplot(grid[2,1])],primary,'BCDE'):\n"
        "        violin(ax,data[rec['key']][1],['Responder','Nonresponder'],[NAVY,WINE],\n"
        "               rec['limits'],rec['comparisons'],rng,title=rec['short_title'],letter=letter)\n"
        "    save(fig,'00_TYROBP_clinical_violin_panels_A-E')\n"
        '\n'
        "    pd.DataFrame([{k:v for k,v in r.items() if k not in ['limits','comparisons','key']} for r in records]).to_csv(\n"
        "        OUT/'figure_panel_map.csv',index=False,encoding='utf-8-sig')\n"
        "    caption='''Publication-style violin plots / 小提琴图说明\n"
        '\n'
        'Style: dark navy and burgundy matching the supplied reference; pale blue marks\n'
        'inactive UC in the three-group panel. White boxes show the interquartile range,\n'
        'black lines the median, whiskers extend to 1.5 IQR, and dots are individual samples.\n'
        'Density curves are truncated at the observed expression range. No data or tests\n'
        'were changed. Sample sizes remain visible, including the n=3 responder group.\n'
        '\n'
        'Minimal main panels A–E:\n'
        'A: GSE75214 colonic Control (11), Inactive UC (23), Active UC (74).\n'
        'B: GSE16879 infliximab, response at weeks 4–6 (8 responders / 16 nonresponders).\n'
        'C: GSE73661 infliximab, response at weeks 4–6 (8 / 15).\n'
        'D: GSE73661 vedolizumab, response at week 6 (6 / 21).\n'
        'E: GSE73661 vedolizumab, response at week 12 (3 / 10).\n'
        '\n'
        'P values in main panels A–E are Holm-adjusted: three pairwise activity contrasts\n'
        'in panel A and four primary baseline response comparisons in panels B–E.\n'
        'The display label is simply P. Exact values remain in the statistical CSV\n'
        'tables. Values below 0.001 are shown as P < 0.001.\n'
        'The separate week-52 vedolizumab figure is secondary and displays raw P=0.13.\n'
        'Do not relabel adjusted P as raw P when assembling the manuscript.\n'
        '\n'
        'Expression scale: RMA log2 intensity for GSE75214 and GSE73661;\n'
        'log2(GCOS globally scaled signal) for GSE16879. Absolute intensity values are\n'
        'not directly comparable between datasets. GSE73661 baseline panels share a\n'
        'common y-range. R/NR labels refer to source-defined response, not randomization.\n'
        'Violin density for n=3 is descriptive only and does not imply a stable population\n'
        'distribution. All individual observations and n values are shown.\n'
        '\n'
        '中文：图A为疾病活动度三组比较；图B–E为治疗前表达与后续应答的比较。\n'
        '主图A–E中的P值均经过Holm校正，图上统一仅标注P。仅保留组名、样本量、表达纵轴和P值；拼图另保留面板字母\n'
        '及简短的数据集/药物/时间标签。第52周单图为次要结果，P为未校正值。\n'
        '白箱为四分位距，黑线为中位数，散点为各样本，曲线仅用于描述分布。\n'
        'IFX=英夫利昔单抗，VDZ=维多珠单抗。使用图时保留此统计说明于图注。\n'
        '\n'
        'Run from the analysis folder: python beautify_figures.py\n'
        "'''\n"
        "    (OUT/'Figure_captions_ZH_EN.txt').write_text(caption,encoding='utf-8-sig')\n"
        "    report=ROOT/'TYROBP_clinical_report.html'\n"
        '    if report.exists():\n'
        "        text=report.read_text(encoding='utf-8')\n"
        "        text=text.replace('figures/01_TYROBP_disease_activity.png','figures/violin_style/01_TYROBP_activity_violin.png')\n"
        "        text=text.replace('figures/02_TYROBP_baseline_response.png','figures/violin_style/02_TYROBP_response_violin.png')\n"
        "        marker='<p>各终点完整的配对轨迹图及患者表位于'\n"
        "        text=text.replace('P<sub>adj</sub>为Holm校正P','主图A–E中的P值均经过Holm校正')\n"
        '        note=\'<p>小提琴图沿用参考图配色和白色箱线样式。主图A–E中的P值均经过Holm校正；精确值见表。<a href="figures/violin_style/00_TYROBP_clinical_violin_panels_A-E.pdf">下载A–E整版图</a> · <a href="figures/violin_style/Figure_captions_ZH_EN.txt">查看中英文图注</a></p>\'\n'
        '        if note not in text:\n'
        '            text=text.replace(marker,note+marker)\n'
        "        report.write_text(text,encoding='utf-8')\n"
        '    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==digest for p,digest in before.items())\n'
        '    # Refresh the release manifest when this display-only script is run alone.\n'
        '    manifest=[]\n'
        "    for path in sorted(ROOT.rglob('*')):\n"
        "        if path.is_file() and path.name!='file_manifest_sha256.csv' and '__pycache__' not in path.parts:\n"
        "            manifest.append({'relative_path':path.relative_to(ROOT).as_posix(),\n"
        "                             'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})\n"
        "    pd.DataFrame(manifest).to_csv(ROOT/'file_manifest_sha256.csv',index=False,encoding='utf-8-sig')\n"
        "    print(json.dumps({'output':str(OUT),'panels':'A–E','individual_and_combined_formats':['PNG 600 dpi','PDF','SVG'],'statistics_unchanged':True},ensure_ascii=False))\n"
        '\n'
        "if __name__=='__main__':\n"
        '    main()\n'
    ),
    'fig09_analyze_gse75214.py': (
        '"""Reproduce the TYROBP disease-activity analysis using public GEO data.\n'
        '\n'
        'Run: python analyze_gse75214.py\n'
        'Inputs are downloaded from NCBI if missing. GEO RMA/log2 values are used as\n'
        'deposited: no second log transform and no across-cohort normalization.\n'
        '"""\n'
        'from pathlib import Path\n'
        'import csv, gzip, hashlib, json, platform, re, shutil, subprocess, urllib.request\n'
        'from collections import Counter\n'
        'import numpy as np\n'
        'import pandas as pd\n'
        'import scipy\n'
        'from scipy import stats\n'
        'from statsmodels.stats.multitest import multipletests\n'
        'import matplotlib\n'
        "matplotlib.use('Agg')\n"
        'import matplotlib.pyplot as plt\n'
        '\n'
        'BASE = Path(__file__).resolve().parent\n'
        "OUT = BASE / 'results'\n"
        'OUT.mkdir(exist_ok=True)\n'
        "MATRIX_URL = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE75nnn/GSE75214/matrix/GSE75214_series_matrix.txt.gz'\n"
        "ANNOT_URL = 'https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL6nnn/GPL6244/annot/GPL6244.annot.gz'\n"
        "GROUPS = ['Control', 'Inactive UC', 'Active UC']\n"
        'SEED = 75214\n'
        '\n'
        'def fetch(url, path):\n'
        '    if path.exists() and path.stat().st_size > 1000:\n'
        '        return\n'
        "    curl = shutil.which('curl.exe') or shutil.which('curl')\n"
        '    if curl:\n'
        "        subprocess.run([curl, '--fail', '--location', '--retry', '3', '--max-time', '900', '--output', str(path), url], check=True)\n"
        '    else:\n'
        "        with urllib.request.urlopen(url, timeout=120) as src, path.open('wb') as dst:\n"
        '            shutil.copyfileobj(src, dst)\n'
        '\n'
        'def split_tsv(line):\n'
        "    return next(csv.reader([line.rstrip('\\r\\n')], delimiter='\\t'))\n"
        '\n'
        'def load_mapping(path):\n'
        '    rows, header = [], None\n'
        "    with gzip.open(path, 'rt', encoding='utf-8') as fh:\n"
        '        for line in fh:\n'
        "            if line.startswith('#') or line.startswith('!') or line.startswith('^'):\n"
        '                continue\n'
        '            parts = split_tsv(line)\n'
        "            if parts[0] == 'ID':\n"
        '                header = parts\n'
        '            elif header and len(parts) == len(header):\n'
        '                row = dict(zip(header, parts))\n'
        "                symbols = {x.strip() for x in re.split(r'///|//|;', row['Gene symbol']) if x.strip()}\n"
        "                if 'TYROBP' in symbols:\n"
        "                    row['unambiguous_TYROBP'] = symbols == {'TYROBP'}\n"
        '                    rows.append(row)\n'
        "    assert rows, 'TYROBP mapping missing'\n"
        '    return pd.DataFrame(rows)\n'
        '\n'
        'def load_series(path, probe_ids):\n'
        '    metadata = []\n'
        '    expression = {}\n'
        '    ids = None\n'
        '    table = False\n'
        "    with gzip.open(path, 'rt', encoding='utf-8') as fh:\n"
        '        for line in fh:\n'
        "            if line.startswith('!series_matrix_table_begin'):\n"
        '                table = True\n'
        '                continue\n'
        "            if line.startswith('!series_matrix_table_end'):\n"
        '                table = False\n'
        '                continue\n'
        '            if not table:\n'
        "                if line.startswith('!Sample_'):\n"
        '                    metadata.append(split_tsv(line))\n'
        '            elif line.startswith(\'"ID_REF"\') or line.startswith(\'ID_REF\'):\n'
        '                ids = split_tsv(line)[1:]\n'
        '            else:\n'
        '                # Stream rather than load the full expression matrix into RAM.\n'
        '                first = line.split(\'\\t\', 1)[0].strip(\'"\')\n'
        '                if first in probe_ids:\n'
        '                    vals = split_tsv(line)\n'
        '                    expression[first] = [float(x) for x in vals[1:]]\n'
        '    assert ids and set(expression) == set(probe_ids)\n'
        '    meta = pd.DataFrame(index=ids)\n'
        '    key_counts = Counter()\n'
        '    for record in metadata:\n'
        "        key, values = record[0].replace('!Sample_', ''), record[1:]\n"
        '        assert len(values) == len(ids)\n'
        "        if key == 'characteristics_ch1':\n"
        "            names = {v.split(':', 1)[0].strip() for v in values}\n"
        '            if len(names) == 1:\n'
        "                key = next(iter(names)).replace(' ', '_')\n"
        "                values = [v.split(':', 1)[1].strip() if ':' in v else v for v in values]\n"
        '        key_counts[key] += 1\n'
        '        if key_counts[key] > 1:\n'
        "            key += '_' + str(key_counts[key])\n"
        '        meta[key] = values\n'
        "    meta.index.name = 'sample_id'\n"
        "    assert list(meta['geo_accession']) == ids\n"
        '    expr = pd.DataFrame(expression, index=ids)\n'
        "    meta['TYROBP_expression_log2'] = expr.mean(axis=1)\n"
        '    for probe in probe_ids:\n'
        "        meta['TYROBP_probe_' + probe] = expr[probe]\n"
        '    return meta\n'
        '\n'
        'def comparison(frame, case, ref, rng):\n'
        "    x = frame.loc[frame.group.eq(case), 'TYROBP_expression_log2'].to_numpy()\n"
        "    y = frame.loc[frame.group.eq(ref), 'TYROBP_expression_log2'].to_numpy()\n"
        "    res = stats.mannwhitneyu(x, y, alternative='two-sided', method='asymptotic', use_continuity=True)\n"
        '    # Nonparametric percentile CIs for descriptive between-group effect sizes.\n'
        '    bx = rng.choice(x, (10000, len(x)), replace=True)\n'
        '    by = rng.choice(y, (10000, len(y)), replace=True)\n'
        '    mean_ci = np.quantile(bx.mean(1)-by.mean(1), [.025, .975])\n'
        '    median_ci = np.quantile(np.median(bx, axis=1)-np.median(by, axis=1), [.025, .975])\n'
        "    return dict(comparison=f'{case} vs {ref}', case_group=case, reference_group=ref,\n"
        '                n_case=len(x), n_reference=len(y), mean_difference_log2=float(x.mean()-y.mean()),\n'
        '                mean_difference_ci_low=float(mean_ci[0]), mean_difference_ci_high=float(mean_ci[1]),\n'
        '                median_difference_log2=float(np.median(x)-np.median(y)),\n'
        '                median_difference_ci_low=float(median_ci[0]), median_difference_ci_high=float(median_ci[1]),\n'
        '                geometric_mean_ratio=float(2**(x.mean()-y.mean())),\n'
        '                cliffs_delta=float(2*res.statistic/(len(x)*len(y))-1),\n'
        '                mannwhitney_u=float(res.statistic), p_value=float(res.pvalue),\n'
        "                test='Two-sided Mann-Whitney U, asymptotic with continuity correction')\n"
        '\n'
        'def main():\n'
        "    matrix = BASE / 'GSE75214_series_matrix.txt.gz'\n"
        "    annot = BASE / 'GPL6244.annot.gz'\n"
        '    fetch(MATRIX_URL, matrix)\n'
        '    fetch(ANNOT_URL, annot)\n'
        '    mapping = load_mapping(annot)\n'
        "    mapping.to_csv(OUT / 'TYROBP_probe_mapping.csv', index=False, encoding='utf-8-sig')\n"
        "    probes = mapping.loc[mapping.unambiguous_TYROBP, 'ID'].tolist()\n"
        "    assert probes, 'No unambiguous TYROBP probe'\n"
        '    meta = load_series(matrix, probes)\n'
        "    meta['included'] = meta.tissue.eq('colon') & meta.disease.isin(['control', 'ulcerative colitis'])\n"
        "    meta['exclusion_reason'] = np.where(meta.included, '', 'Crohn disease and/or ileum biopsy')\n"
        "    meta['group'] = ''\n"
        "    meta.loc[meta.included & meta.disease.eq('control'), 'group'] = 'Control'\n"
        "    meta.loc[meta.included & meta.disease.eq('ulcerative colitis') & meta.disease_activity.eq('inactive'), 'group'] = 'Inactive UC'\n"
        "    meta.loc[meta.included & meta.disease.eq('ulcerative colitis') & meta.disease_activity.eq('active'), 'group'] = 'Active UC'\n"
        "    assert meta.loc[meta.included, 'group'].ne('').all()\n"
        '    chosen = meta.loc[meta.included].copy()\n'
        "    assert chosen.group.value_counts().to_dict() == {'Active UC': 74, 'Inactive UC': 23, 'Control': 11}\n"
        '    assert len(chosen) == 108 and not chosen.index.duplicated().any()\n'
        "    meta.to_csv(OUT / 'all_samples_metadata.csv', encoding='utf-8-sig')\n"
        '    # Compact table for audit, plotting and central-report generation.\n'
        "    cols = ['title', 'tissue', 'disease', 'disease_activity', 'group', 'TYROBP_expression_log2'] + ['TYROBP_probe_' + p for p in probes]\n"
        "    chosen[cols].to_csv(OUT / 'sample_expression.csv', encoding='utf-8-sig')\n"
        '    descriptives = []\n'
        '    for g in GROUPS:\n'
        "        vals = chosen.loc[chosen.group.eq(g), 'TYROBP_expression_log2']\n"
        '        descriptives.append(dict(group=g, n=len(vals), mean=float(vals.mean()), sd=float(vals.std()),\n'
        '                                 median=float(vals.median()), q1=float(vals.quantile(.25)), q3=float(vals.quantile(.75)),\n'
        '                                 minimum=float(vals.min()), maximum=float(vals.max())))\n'
        "    pd.DataFrame(descriptives).to_csv(OUT / 'group_summary.csv', index=False, encoding='utf-8-sig')\n"
        "    kw = stats.kruskal(*[chosen.loc[chosen.group.eq(g), 'TYROBP_expression_log2'] for g in GROUPS])\n"
        '    rng = np.random.default_rng(SEED)\n'
        "    comparisons = [comparison(chosen, 'Inactive UC', 'Control', rng),\n"
        "                   comparison(chosen, 'Active UC', 'Control', rng),\n"
        "                   comparison(chosen, 'Active UC', 'Inactive UC', rng)]\n"
        "    adj = multipletests([d['p_value'] for d in comparisons], method='holm')[1]\n"
        '    for d, p in zip(comparisons, adj):\n'
        "        d['p_holm'] = float(p)\n"
        "    pd.DataFrame(comparisons).to_csv(OUT / 'pairwise_comparisons.csv', index=False, encoding='utf-8-sig')\n"
        "    report = dict(dataset='GSE75214', platform='GPL6244', gene='TYROBP', probe_ids=probes,\n"
        '                  counts=chosen.group.value_counts().to_dict(), group_summary=descriptives,\n'
        "                  overall_test=dict(test='Kruskal-Wallis', statistic=float(kw.statistic), p_value=float(kw.pvalue)),\n"
        '                  comparisons=comparisons,\n'
        '                  preprocessing=str(meta.data_processing.iloc[0]),\n'
        "                  activity_definition='Endoscopically assessed, using deposited active/inactive labels; not necessarily histological remission.',\n"
        "                  multiplicity='Holm adjustment across the three disease-activity pairwise comparisons.',\n"
        "                  limitations=['Cross-sectional observational cohort; no causal interpretation.',\n"
        "                               'Inactive means deposited endoscopically inactive disease; detailed remission thresholds are not present in sample metadata.',\n"
        "                               'Age, medication and other clinical covariates are not provided in the deposited per-sample matrix metadata and are not adjusted.',\n"
        "                               'No independent diagnostic validation or diagnostic cutoff was established.'],\n"
        '                  sources=[dict(url=MATRIX_URL, file=matrix.name, sha256=hashlib.sha256(matrix.read_bytes()).hexdigest()),\n'
        '                           dict(url=ANNOT_URL, file=annot.name, sha256=hashlib.sha256(annot.read_bytes()).hexdigest()),\n'
        "                           dict(url='https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE75214')],\n"
        '                  software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__),\n'
        '                  random_seed=SEED, bootstrap_resamples=10000)\n'
        "    (OUT / 'analysis_stats.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')\n"
        "    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'pdf.fonttype':42, 'svg.fonttype':'none'})\n"
        '    fig, ax = plt.subplots(figsize=(7.0, 5.8))\n'
        "    colors = ['#607d9a', '#58a89b', '#ba3c5a']\n"
        "    vals = [chosen.loc[chosen.group.eq(g), 'TYROBP_expression_log2'].to_numpy() for g in GROUPS]\n"
        '    bp = ax.boxplot(vals, positions=[1,2,3], widths=.52, patch_artist=True, showfliers=False,\n'
        "                    medianprops={'color':'black','linewidth':1.6}, whiskerprops={'linewidth':1}, capprops={'linewidth':1})\n"
        '    jit = np.random.default_rng(SEED)\n'
        "    for k, (v, c, b) in enumerate(zip(vals, colors, bp['boxes']), start=1):\n"
        '        b.set(facecolor=c, alpha=.25, edgecolor=c)\n'
        "        ax.scatter(k+jit.uniform(-.16,.16,len(v)), v, s=29, facecolors=c, edgecolors='white', linewidths=.4, alpha=.85, zorder=3)\n"
        '    lower = min(min(v) for v in vals)\n'
        '    upper = max(max(v) for v in vals)\n'
        "    for i, (x1, x2, p) in enumerate([(1,2,comparisons[0]['p_holm']), (2,3,comparisons[2]['p_holm']), (1,3,comparisons[1]['p_holm'])]):\n"
        '        y = upper + .25 + .42*i\n'
        "        ax.plot([x1,x1,x2,x2], [y,y+.07,y+.07,y], color='#30343b', linewidth=.8)\n"
        "        lab = f'Holm P = {p:.3f}' if p >= .001 else f'Holm P = {p:.2e}'\n"
        "        ax.text((x1+x2)/2, y+.11, lab, ha='center', va='bottom', fontsize=9)\n"
        "    ax.set(xticks=[1,2,3], xticklabels=[f'{g}\\n(n = {len(v)})' for g,v in zip(GROUPS,vals)],\n"
        "           ylabel='TYROBP expression (RMA, log2)', ylim=(lower-.25, upper+1.7),\n"
        "           title='GSE75214: TYROBP across UC activity groups')\n"
        "    ax.spines[['top','right']].set_visible(False)\n"
        "    ax.grid(axis='y', alpha=.15)\n"
        "    fig.text(.02,.015, f'Kruskal-Wallis P = {kw.pvalue:.2e}; two-sided pairwise Mann-Whitney tests, Holm-adjusted.', fontsize=8)\n"
        '    fig.tight_layout(rect=[0,.045,1,1])\n'
        "    for ext in ['png','pdf','svg']:\n"
        "        fig.savefig(OUT / f'GSE75214_TYROBP_activity.{ext}', dpi=300, bbox_inches='tight')\n"
        '    plt.close(fig)\n'
        "    print(json.dumps({'counts':report['counts'], 'probe_ids':probes, 'overall_test':report['overall_test'], 'comparisons':comparisons}, indent=2))\n"
        '\n'
        "if __name__ == '__main__':\n"
        '    main()\n'
    ),
    'fig09_analyze_gse16879.py': (
        '"""Reproducible exploratory TYROBP analysis, GSE16879.\n'
        '\n'
        'Primary: pretreatment UC responders vs nonresponders only.\n'
        'Secondary: explicitly paired UC before/after first infliximab treatment.\n'
        "Input is GEO's submitter-processed GCOS globally scaled signal, not CEL/RMA.\n"
        'No outcome-driven probe selection, cutoff selection, or ROC direction reversal.\n'
        'Run: python analyze_gse16879.py --bootstrap 10000\n'
        '"""\n'
        'from __future__ import annotations\n'
        'import argparse\n'
        'import csv\n'
        'import gzip\n'
        'import hashlib\n'
        'import json\n'
        'import platform\n'
        'import re\n'
        'import time\n'
        'import urllib.request\n'
        'from pathlib import Path\n'
        '\n'
        'import numpy as np\n'
        'import pandas as pd\n'
        'import scipy\n'
        'from scipy import stats\n'
        'import matplotlib\n'
        'matplotlib.use("Agg")\n'
        'import matplotlib.pyplot as plt\n'
        'from sklearn.metrics import roc_curve, roc_auc_score\n'
        'import sklearn\n'
        '\n'
        'ROOT = Path(__file__).resolve().parent\n'
        'URLS = {\n'
        '    "series": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE16879&targ=self&form=text&view=full",\n'
        '    "matrix": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE16nnn/GSE16879/matrix/GSE16879_series_matrix.txt.gz",\n'
        '    "annotation": "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz",\n'
        '    "publication": "https://pubmed.ncbi.nlm.nih.gov/19956723/",\n'
        '}\n'
        '\n'
        'def download(url, filename):\n'
        '    path = ROOT / filename\n'
        '    if path.exists() and path.stat().st_size > 0:\n'
        '        return path\n'
        '    for attempt in range(4):\n'
        '        try:\n'
        '            urllib.request.urlretrieve(url, path)\n'
        '            return path\n'
        '        except Exception:\n'
        '            if path.exists(): path.unlink()\n'
        '            if attempt == 3: raise\n'
        '            time.sleep(2 * (attempt + 1))\n'
        '\n'
        'def parse_matrix(path):\n'
        '    metadata = []\n'
        '    gene_rows = {}\n'
        '    with gzip.open(path, "rt", encoding="utf-8") as handle:\n'
        '        inside = False\n'
        '        for line in handle:\n'
        '            if not line.strip(): continue\n'
        '            if line.startswith("!series_matrix_table_begin"):\n'
        '                inside = True\n'
        '                continue\n'
        '            if line.startswith("!series_matrix_table_end"): break\n'
        '            row = next(csv.reader([line], delimiter="\\t"))\n'
        '            if not inside:\n'
        '                if row[0].startswith("!Sample_"): metadata.append(row)\n'
        '            elif row[0] == "ID_REF":\n'
        '                sample_ids = row[1:]\n'
        '            elif row[0] in PROBES:\n'
        '                gene_rows[row[0]] = np.asarray(row[1:], dtype=float)\n'
        '    fields = {}\n'
        '    chars = []\n'
        '    for row in metadata:\n'
        '        key = row[0].removeprefix("!Sample_")\n'
        '        if key == "characteristics_ch1": chars.append(row[1:])\n'
        '        elif key in ["title", "geo_accession", "source_name_ch1", "description", "data_processing"]:\n'
        '            fields[key] = row[1:]\n'
        '    df = pd.DataFrame(fields)\n'
        '    for i, values in enumerate(chars):\n'
        '        names, vals = zip(*(v.split(": ", 1) if ": " in v else (f"characteristic_{i}", v) for v in values))\n'
        '        assert len(set(names)) == 1\n'
        '        df[names[0]] = vals\n'
        '    assert df.geo_accession.tolist() == sample_ids\n'
        '    for probe, values in gene_rows.items(): df[probe] = values\n'
        '    return df\n'
        '\n'
        'def scalar(x):\n'
        '    return x.item() if isinstance(x, np.generic) else x\n'
        '\n'
        'def main():\n'
        '    parser = argparse.ArgumentParser()\n'
        '    parser.add_argument("--bootstrap", type=int, default=10000)\n'
        '    args = parser.parse_args()\n'
        '    rng = np.random.default_rng(16879)\n'
        '    matrix = download(URLS["matrix"], "GSE16879_series_matrix.txt.gz")\n'
        '    annot = download(URLS["annotation"], "GPL570.annot.gz")\n'
        '    # Verify exact gene-symbol mapping rather than assuming a published probe ID.\n'
        '    matched = []\n'
        '    with gzip.open(annot, "rt", encoding="utf-8") as handle:\n'
        '        for line in handle:\n'
        '            if line.startswith("ID\\t"):\n'
        '                columns = line.rstrip("\\r\\n").split("\\t")\n'
        '                break\n'
        '        for row in csv.DictReader(handle, fieldnames=columns, delimiter="\\t"):\n'
        '            if row.get("Gene symbol", "") == "TYROBP": matched.append(row)\n'
        '    global PROBES\n'
        '    PROBES = [row["ID"] for row in matched]\n'
        '    assert PROBES == ["204122_at"], f"Unexpected TYROBP probes: {PROBES}"\n'
        '    pd.DataFrame(matched).to_csv(ROOT / "TYROBP_probe_annotation.csv", index=False)\n'
        '    df = parse_matrix(matrix)\n'
        '    df["TYROBP_signal"] = df[PROBES].mean(axis=1)\n'
        '    assert (df.TYROBP_signal > 0).all()\n'
        '    df["TYROBP_log2"] = np.log2(df.TYROBP_signal)\n'
        '    df["included_UC"] = df["disease"].eq("UC") & df["tissue"].eq("Colon")\n'
        '    df["exclusion_reason"] = np.where(df.included_UC, "", "Not UC colonic biopsy (CD or healthy control)")\n'
        '    uc = df[df.included_UC].copy()\n'
        '    parsed = uc.title.str.extract(r"^(UC(?:NR|R)\\d+)_(beforeT|afterT)$")\n'
        '    assert parsed.notna().all().all()\n'
        '    uc["patient_id"] = parsed[0]\n'
        '    uc["timepoint"] = parsed[1].map({"beforeT": "Baseline", "afterT": "Post-treatment"})\n'
        '    uc["response_group"] = uc["response to infliximab"].map({"Yes": "Responder", "No": "Nonresponder"})\n'
        '    assert uc.response_group.notna().all()\n'
        '    assert uc.groupby("patient_id").response_group.nunique().eq(1).all()\n'
        '    assert uc.groupby(["patient_id", "timepoint"]).size().eq(1).all()\n'
        '    df.to_csv(ROOT / "all_sample_metadata_and_TYROBP.csv", index=False)\n'
        '    uc.to_csv(ROOT / "UC_sample_metadata_and_TYROBP.csv", index=False)\n'
        '    baseline = uc[uc.timepoint.eq("Baseline")].copy()\n'
        '    baseline.to_csv(ROOT / "UC_baseline_TYROBP.csv", index=False)\n'
        '    r = baseline.loc[baseline.response_group.eq("Responder"), "TYROBP_log2"].to_numpy()\n'
        '    nr = baseline.loc[baseline.response_group.eq("Nonresponder"), "TYROBP_log2"].to_numpy()\n'
        '    assert len(r) == 8 and len(nr) == 16\n'
        '    mann = stats.mannwhitneyu(nr, r, alternative="two-sided", method="auto")\n'
        '    auc = mann.statistic / (len(nr) * len(r))\n'
        '    auc_draws = []\n'
        '    diff_draws = []\n'
        '    for _ in range(args.bootstrap):\n'
        '        rb = rng.choice(r, len(r), replace=True)\n'
        '        nb = rng.choice(nr, len(nr), replace=True)\n'
        '        # Probability that a randomly selected NR has higher signal than an R.\n'
        '        auc_draws.append(np.mean(nb[:, None] > rb[None, :]) + 0.5 * np.mean(nb[:, None] == rb[None, :]))\n'
        '        diff_draws.append(nb.mean() - rb.mean())\n'
        '    paired = uc.pivot(index=["patient_id", "response_group"], columns="timepoint", values="TYROBP_log2").reset_index()\n'
        '    assert paired[["Baseline", "Post-treatment"]].notna().all().all()\n'
        '    paired["change_post_minus_baseline"] = paired["Post-treatment"] - paired["Baseline"]\n'
        '    paired.to_csv(ROOT / "UC_paired_TYROBP.csv", index=False)\n'
        '    paired_stats = []\n'
        '    for group in ["Responder", "Nonresponder"]:\n'
        '        sub = paired[paired.response_group.eq(group)]\n'
        '        wt = stats.wilcoxon(sub["Post-treatment"], sub.Baseline, alternative="two-sided", method="auto")\n'
        '        paired_stats.append({"analysis": f"{group}: paired post-treatment vs baseline", "n_pairs": len(sub), "median_change_log2": sub.change_post_minus_baseline.median(), "mean_change_log2": sub.change_post_minus_baseline.mean(), "statistic": wt.statistic, "p_value": wt.pvalue, "method": "Two-sided paired Wilcoxon signed-rank"})\n'
        '    dr = paired.loc[paired.response_group.eq("Responder"), "change_post_minus_baseline"]\n'
        '    dn = paired.loc[paired.response_group.eq("Nonresponder"), "change_post_minus_baseline"]\n'
        '    dm = stats.mannwhitneyu(dn, dr, alternative="two-sided", method="auto")\n'
        '    paired_stats.append({"analysis": "Post-minus-baseline change: nonresponders vs responders", "n_pairs": len(paired), "median_change_log2": None, "mean_change_log2": dn.mean()-dr.mean(), "statistic": dm.statistic, "p_value": dm.pvalue, "method": "Two-sided Mann-Whitney U"})\n'
        '    pd.DataFrame(paired_stats).to_csv(ROOT / "secondary_longitudinal_statistics.csv", index=False)\n'
        '    summary = {\n'
        '        "dataset": "GSE16879", "platform": "GPL570", "gene": "TYROBP", "probes": PROBES,\n'
        '        "treatment": "Infliximab", "response_definition": "Submitter-defined response based on endoscopic and histologic findings 4-6 weeks after first infliximab treatment",\n'
        '        "preprocessing": "Submitter GCOS signal globally scaled to average 150 per array; log2(signal), without adding a pseudocount; no CEL reprocessing or between-cohort normalization",\n'
        '        "primary_analysis": {\n'
        '            "population": "UC, colon, before first infliximab treatment only", "n_responders": len(r), "n_nonresponders": len(nr),\n'
        '            "median_responders_log2": np.median(r), "median_nonresponders_log2": np.median(nr),\n'
        '            "mean_responders_log2": r.mean(), "mean_nonresponders_log2": nr.mean(),\n'
        '            "mean_difference_nonresponders_minus_responders_log2": nr.mean()-r.mean(),\n'
        '            "mean_difference_bootstrap_95CI": np.quantile(diff_draws, [0.025, 0.975]).tolist(),\n'
        '            "test": "Two-sided Mann-Whitney U; scipy method=auto", "U_nonresponders_vs_responders": mann.statistic, "p_value": mann.pvalue,\n'
        '            "auc_fixed_direction_high_expression_predicts_nonresponse": auc,\n'
        '            "auc_stratified_bootstrap_95CI": np.quantile(auc_draws, [0.025, 0.975]).tolist(),\n'
        '            "bootstrap_iterations": args.bootstrap, "bootstrap_seed": 16879,\n'
        '        },\n'
        '        "secondary_longitudinal": paired_stats,\n'
        '        "sources": URLS,\n'
        '        "source_sha256": {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in [matrix, annot]},\n'
        '        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__, "sklearn": sklearn.__version__},\n'
        '        "limitations": ["Exploratory single-gene reanalysis; no clinical covariate adjustment, locked diagnostic threshold, or prospective validation", "Small baseline responder group (n=8)", "Response is submitter-defined endoscopic/histologic response, not necessarily equivalent to clinical remission", "Post-treatment changes are pharmacodynamic associations and must not be described as pretreatment prediction", "Potential subject overlap with GSE73661 requires verification before claiming independent replication", "Macrophage/inflammatory-cell composition and inflammation severity are not controlled here"],\n'
        '    }\n'
        '    (ROOT / "results_GSE16879.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=scalar), encoding="utf-8")\n'
        '    # Scientific figures: shared log2-intensity axis, individual subjects shown.\n'
        '    blue, red = "#24739B", "#B44758"\n'
        '    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))\n'
        '    ax = axes[0]\n'
        '    box = ax.boxplot([r, nr], positions=[0, 1], widths=0.5, showfliers=False, patch_artist=True)\n'
        '    for patch, color in zip(box["boxes"], [blue, red]): patch.set_facecolor(color); patch.set_alpha(0.23)\n'
        '    for i, (values, color) in enumerate(zip([r, nr], [blue, red])):\n'
        '        ax.scatter(i + rng.uniform(-.11, .11, len(values)), values, color=color, s=34, edgecolor="white", linewidth=.4, zorder=3)\n'
        '    ax.set_xticks([0, 1], ["Responders\\n(n = 8)", "Nonresponders\\n(n = 16)"])\n'
        '    ax.set_ylabel("Baseline TYROBP expression (log$_2$ signal)")\n'
        '    ax.set_title(f"GSE16879: infliximab\\nMann–Whitney P = {mann.pvalue:.4g}")\n'
        '    ax = axes[1]\n'
        '    y = baseline.response_group.eq("Nonresponder").astype(int)\n'
        '    fpr, tpr, _ = roc_curve(y, baseline.TYROBP_log2)\n'
        '    ax.plot(fpr, tpr, color=red, lw=2)\n'
        '    ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1)\n'
        '    lo, hi = np.quantile(auc_draws, [.025, .975])\n'
        '    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.02))\n'
        '    ax.set_title(f"ROC direction: higher TYROBP → nonresponse\\nAUC = {auc:.3f} (95% CI {lo:.3f}–{hi:.3f})")\n'
        '    for ax in axes: ax.spines[["top", "right"]].set_visible(False)\n'
        '    fig.tight_layout()\n'
        '    fig.savefig(ROOT / "GSE16879_baseline_response.png", dpi=300)\n'
        '    fig.savefig(ROOT / "GSE16879_baseline_response.pdf")\n'
        '    plt.close(fig)\n'
        '    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)\n'
        '    for ax, group, color, ps in zip(axes, ["Responder", "Nonresponder"], [blue, red], paired_stats):\n'
        '        sub = paired[paired.response_group.eq(group)]\n'
        '        for _, row in sub.iterrows():\n'
        '            ax.plot([0, 1], [row.Baseline, row["Post-treatment"]], color=color, alpha=.5, marker="o", ms=3)\n'
        '        ax.set_xticks([0, 1], ["Baseline", "4–6 weeks"])\n'
        '        ax.set_title(f"{group}s (n = {len(sub)})\\nPaired Wilcoxon P = {ps[\'p_value\']:.4g} (raw)")\n'
        '        ax.spines[["top", "right"]].set_visible(False)\n'
        '    axes[0].set_ylabel("TYROBP expression (log$_2$ signal)")\n'
        '    fig.suptitle(f"GSE16879: within-patient changes after infliximab\\nBetween-response-group change comparison: P = {dm.pvalue:.4g}", y=1.1)\n'
        '    fig.tight_layout()\n'
        '    fig.savefig(ROOT / "GSE16879_paired_change.png", dpi=300, bbox_inches="tight")\n'
        '    fig.savefig(ROOT / "GSE16879_paired_change.pdf", bbox_inches="tight")\n'
        '    plt.close(fig)\n'
        '    print(json.dumps(summary, indent=2, ensure_ascii=False, default=scalar))\n'
        '\n'
        'if __name__ == "__main__": main()\n'
    ),
    'fig09_analyze_gse73661.py': (
        '"""Reproducible single-gene analysis of GSE73661, using official GEO metadata.\n'
        '\n'
        'Run: python analyze_gse73661.py\n'
        'Processed RMA log2 values are used without an additional log transformation.\n'
        'No cutpoint, gene selection, AUC direction reversal, or outcome imputation is used.\n'
        '"""\n'
        'from __future__ import annotations\n'
        '\n'
        'import csv\n'
        'import gzip\n'
        'import io\n'
        'import json\n'
        'import platform\n'
        'import re\n'
        'import subprocess\n'
        'import sys\n'
        'from pathlib import Path\n'
        '\n'
        'import matplotlib\n'
        'matplotlib.use("Agg")\n'
        'import matplotlib.pyplot as plt\n'
        'import numpy as np\n'
        'import pandas as pd\n'
        'import scipy\n'
        'from scipy import stats\n'
        'from sklearn.metrics import roc_auc_score, roc_curve\n'
        'import sklearn\n'
        '\n'
        'plt.rcParams.update({"pdf.fonttype": 42, "font.size": 10})\n'
        '\n'
        'BASE = Path(__file__).resolve().parent\n'
        'OUT = BASE / "results"\n'
        'FIG = BASE / "figures"\n'
        'SEED = 73661\n'
        'BOOTSTRAPS = 10000\n'
        'URLS = {\n'
        '    "GSE73661_metadata.txt": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE73661&targ=self&form=text&view=full",\n'
        '    "GSE73661_samples_metadata.txt": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE73661&targ=gsm&form=text&view=brief",\n'
        '    "series_matrix_download.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE73nnn/GSE73661/matrix/GSE73661_series_matrix.txt.gz",\n'
        '    "GPL6244_download.annot.gz": "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL6nnn/GPL6244/annot/GPL6244.annot.gz",\n'
        '}\n'
        '\n'
        '\n'
        'def fetch_if_missing():\n'
        '    """PowerShell respects the Windows proxy/certificate configuration."""\n'
        '    for name, url in URLS.items():\n'
        '        path = BASE / name\n'
        '        if path.exists() and path.stat().st_size > 1000:\n'
        '            continue\n'
        '        dest = str(path).replace("\'", "\'\'")\n'
        '        command = f"Invoke-WebRequest -Uri \'{url}\' -OutFile \'{dest}\' -UseBasicParsing -TimeoutSec 180"\n'
        '        subprocess.run(["powershell", "-NoProfile", "-Command", command], check=True, timeout=210)\n'
        '\n'
        '\n'
        'def parse_metadata():\n'
        '    text = (BASE / "GSE73661_samples_metadata.txt").read_text(encoding="utf-8-sig")\n'
        '    rows = []\n'
        '    for block in re.split(r"(?m)^\\^SAMPLE = ", text)[1:]:\n'
        '        row = {"gsm": block.splitlines()[0].strip()}\n'
        '        for line in block.splitlines()[1:]:\n'
        '            if " = " not in line:\n'
        '                continue\n'
        '            key, value = line.split(" = ", 1)\n'
        '            if key == "!Sample_characteristics_ch1":\n'
        '                label, sep, val = value.partition(": ")\n'
        '                row[label] = val if sep else value\n'
        '            elif key in ("!Sample_title", "!Sample_source_name_ch1", "!Sample_description", "!Sample_data_processing"):\n'
        '                row[key.replace("!Sample_", "")] = value\n'
        '        rows.append(row)\n'
        '    df = pd.DataFrame(rows).rename(columns={\n'
        '        "study individual number": "patient_id",\n'
        '        "week (w)": "week",\n'
        '        "induction therapy_maintenance therapy": "regimen",\n'
        '        "mayo endoscopic subscore": "mayo_text",\n'
        '    })\n'
        '    df["patient_id"] = df["patient_id"].astype(str)\n'
        '    df["mayo"] = pd.to_numeric(df["mayo_text"], errors="coerce")\n'
        '    # Explicit structured characteristics take precedence over free-text titles.\n'
        '    df["title_week"] = df["title"].str.extract(r"\\b(W\\d+(?:/\\d+)?)(?=[_\\s)]|$)")[0].replace({"W4/6": "W4_W6"})\n'
        '    df["week_title_conflict"] = df["title_week"].notna() & (df["title_week"] != df["week"])\n'
        '    assert len(df) == 178 and df["gsm"].is_unique\n'
        '    return df\n'
        '\n'
        '\n'
        'def tyrobp_probe():\n'
        '    with gzip.open(BASE / "GPL6244_download.annot.gz", "rt", encoding="utf-8-sig") as f:\n'
        '        lines = [line for line in f if not line.startswith(("#", "!", "^")) and line.strip()]\n'
        '    annotation = pd.read_csv(io.StringIO("".join(lines)), sep="\\t", dtype=str, keep_default_na=False)\n'
        '    exact = annotation[annotation["Gene symbol"].eq("TYROBP")]\n'
        '    exact.to_csv(OUT / "TYROBP_probe_annotation.csv", index=False, encoding="utf-8-sig")\n'
        '    probes = exact["ID"].tolist()\n'
        '    assert probes == ["8036224"], f"Unexpected TYROBP annotation: {probes}"\n'
        '    return probes\n'
        '\n'
        '\n'
        'def read_expression(probes):\n'
        '    wanted = set(probes)\n'
        '    data = {}\n'
        '    with gzip.open(BASE / "series_matrix_download.gz", "rt", encoding="utf-8-sig") as f:\n'
        '        table = False\n'
        '        for line in f:\n'
        '            if line.startswith("!series_matrix_table_begin"):\n'
        '                columns = next(csv.reader([next(f)], delimiter="\\t"))[1:]\n'
        '                table = True\n'
        '                continue\n'
        '            if line.startswith("!series_matrix_table_end"):\n'
        '                break\n'
        '            if table:\n'
        '                fields = next(csv.reader([line], delimiter="\\t"))\n'
        '                if fields[0] in wanted:\n'
        '                    data[fields[0]] = np.asarray(fields[1:], dtype=float)\n'
        '    assert set(data) == wanted\n'
        '    # Only one uniquely mapped transcript cluster exists; no probe selection by result.\n'
        '    values = np.mean(np.stack([data[x] for x in probes]), axis=0)\n'
        '    return pd.DataFrame({"gsm": columns, "TYROBP_log2_RMA": values})\n'
        '\n'
        '\n'
        'def bootstrap_stats(nonresponder, responder, rng):\n'
        '    nr, r = np.asarray(nonresponder), np.asarray(responder)\n'
        '    delta = np.empty(BOOTSTRAPS)\n'
        '    auc = np.empty(BOOTSTRAPS)\n'
        '    # Fixed direction: higher TYROBP predicts NONRESPONSE (positive label = 1).\n'
        '    for i in range(BOOTSTRAPS):\n'
        '        nr_b = rng.choice(nr, len(nr), replace=True)\n'
        '        r_b = rng.choice(r, len(r), replace=True)\n'
        '        delta[i] = nr_b.mean() - r_b.mean()\n'
        '        auc[i] = (np.greater.outer(nr_b, r_b).sum() + 0.5 * np.equal.outer(nr_b, r_b).sum()) / (len(nr) * len(r))\n'
        '    return tuple(np.quantile(delta, [0.025, 0.975])), tuple(np.quantile(auc, [0.025, 0.975]))\n'
        '\n'
        '\n'
        'def save_figure(fig, stem):\n'
        '    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")\n'
        '    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")\n'
        '    plt.close(fig)\n'
        '\n'
        '\n'
        'ENDPOINTS = [\n'
        '    {"endpoint": "IFX_W4_W6", "treatment": "Infliximab", "week": "W4_W6", "regimens": ["IFX"], "family": "baseline_primary"},\n'
        '    {"endpoint": "VDZ_W6", "treatment": "Vedolizumab", "week": "W6", "regimens": ["vdz_vdz4w", "vdz_vdz8w", "vdz_plac"], "family": "baseline_primary"},\n'
        '    {"endpoint": "VDZ_W12", "treatment": "Vedolizumab", "week": "W12", "regimens": ["vdz4w"], "family": "baseline_primary"},\n'
        '    {"endpoint": "VDZ_W52", "treatment": "Vedolizumab", "week": "W52", "regimens": ["vdz_vdz4w", "vdz_vdz8w", "vdz4w"], "family": "baseline_secondary"},\n'
        ']\n'
        '\n'
        '\n'
        'def run_endpoint(df, cfg, rng):\n'
        '    eligible = df[df["regimen"].isin(cfg["regimens"])].copy()\n'
        '    baseline = eligible[eligible["week"].eq("W0")].copy()\n'
        '    followup = eligible[eligible["week"].eq(cfg["week"]) & eligible["mayo"].notna()].copy()\n'
        '    assert not baseline.duplicated(["patient_id", "regimen"]).any()\n'
        '    assert not followup.duplicated(["patient_id", "regimen"]).any()\n'
        '    pairs = baseline.merge(followup, on=["patient_id", "regimen"], suffixes=("_baseline", "_followup"), validate="one_to_one")\n'
        '    pairs["endpoint"] = cfg["endpoint"]\n'
        '    pairs["response"] = np.where(pairs["mayo_followup"] <= 1, "Responder", "Nonresponder")\n'
        '    pairs["change_log2"] = pairs["TYROBP_log2_RMA_followup"] - pairs["TYROBP_log2_RMA_baseline"]\n'
        '    pairs.to_csv(OUT / f"{cfg[\'endpoint\']}_paired_samples.csv", index=False, encoding="utf-8-sig")\n'
        '    baseline["endpoint"] = cfg["endpoint"]\n'
        '    baseline["included"] = baseline["gsm"].isin(pairs["gsm_baseline"])\n'
        '    baseline["exclusion_reason"] = np.where(baseline["included"], "", "No observed endpoint at the specified week; not imputed")\n'
        '    baseline.to_csv(OUT / f"{cfg[\'endpoint\']}_baseline_eligibility.csv", index=False, encoding="utf-8-sig")\n'
        '    r = pairs.loc[pairs.response.eq("Responder"), "TYROBP_log2_RMA_baseline"].to_numpy()\n'
        '    nr = pairs.loc[pairs.response.eq("Nonresponder"), "TYROBP_log2_RMA_baseline"].to_numpy()\n'
        '    mw = stats.mannwhitneyu(nr, r, alternative="two-sided", method="exact")\n'
        '    labels = pairs.response.eq("Nonresponder").astype(int).to_numpy()\n'
        '    expr = pairs.TYROBP_log2_RMA_baseline.to_numpy()\n'
        '    auc = roc_auc_score(labels, expr)\n'
        '    diff_ci, auc_ci = bootstrap_stats(nr, r, rng)\n'
        '    result = {"dataset": "GSE73661", "endpoint": cfg["endpoint"], "family": cfg["family"], "treatment": cfg["treatment"],\n'
        '              "response_definition": "Observed endpoint Mayo endoscopic subscore <=1; >1 nonresponse", "followup_week": cfg["week"],\n'
        '              "n_eligible_baseline": len(baseline), "n_paired": len(pairs), "n_responder": len(r), "n_nonresponder": len(nr),\n'
        '              "baseline_responder_mean": float(r.mean()), "baseline_nonresponder_mean": float(nr.mean()),\n'
        '              "baseline_responder_median": float(np.median(r)), "baseline_nonresponder_median": float(np.median(nr)),\n'
        '              "mean_difference_NR_minus_R": float(nr.mean()-r.mean()), "mean_difference_boot95_low": float(diff_ci[0]), "mean_difference_boot95_high": float(diff_ci[1]),\n'
        '              "mannwhitney_U_NR": float(mw.statistic), "p_value": float(mw.pvalue), "test": "Two-sided exact Mann-Whitney U",\n'
        '              "auc_higher_predicts_nonresponse": float(auc), "auc_boot95_low": float(auc_ci[0]), "auc_boot95_high": float(auc_ci[1]),\n'
        '              "bootstrap_iterations": BOOTSTRAPS, "probe_id": "8036224", "expression_scale": "RMA log2; source processed data; no further log transform"}\n'
        '    for status in ["Responder", "Nonresponder"]:\n'
        '        sub = pairs[pairs.response.eq(status)]\n'
        '        changes = sub.change_log2.to_numpy()\n'
        '        w = stats.wilcoxon(changes, alternative="two-sided", method="exact")\n'
        '        result[f"paired_{status}_n"] = len(sub)\n'
        '        result[f"paired_{status}_median_change_log2"] = float(np.median(changes))\n'
        '        result[f"paired_{status}_mean_change_log2"] = float(np.mean(changes))\n'
        '        result[f"paired_{status}_wilcoxon_p"] = float(w.pvalue)\n'
        '    dr = pairs.loc[pairs.response.eq("Responder"), "change_log2"]\n'
        '    dnr = pairs.loc[pairs.response.eq("Nonresponder"), "change_log2"]\n'
        '    result["change_difference_MW_p"] = float(stats.mannwhitneyu(dnr, dr, alternative="two-sided", method="exact").pvalue)\n'
        '    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))\n'
        '    colors = ["#2c7fb8", "#b2182b"]\n'
        '    axes[0].boxplot([r, nr], tick_labels=[f"Responder\\n(n={len(r)})", f"Nonresponder\\n(n={len(nr)})"], widths=.45, showfliers=False)\n'
        '    for i, (values, color) in enumerate(zip([r, nr], colors), 1):\n'
        '        axes[0].scatter(i+rng.uniform(-.09,.09,len(values)), values, c=color, s=32, edgecolors="white", linewidths=.4)\n'
        '    axes[0].set_ylabel("Baseline TYROBP expression (RMA log2)")\n'
        '    axes[0].set_title(f"{cfg[\'treatment\']}, {cfg[\'week\']} outcome\\nUnadjusted P = {mw.pvalue:.4g}")\n'
        '    fpr, tpr, _ = roc_curve(labels, expr)\n'
        '    axes[1].plot(fpr, tpr, color="#653b85", lw=2, label=f"AUC = {auc:.3f}\\n95% bootstrap CI {auc_ci[0]:.3f}–{auc_ci[1]:.3f}")\n'
        '    axes[1].plot([0,1],[0,1],"--",color="grey",lw=1)\n'
        '    axes[1].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC orientation: higher = nonresponse", xlim=(0,1), ylim=(0,1))\n'
        '    axes[1].legend(loc="lower right", frameon=False, fontsize=8)\n'
        '    for ax in axes:\n'
        '        ax.spines[["top","right"]].set_visible(False)\n'
        '    fig.tight_layout()\n'
        '    save_figure(fig, f"{cfg[\'endpoint\']}_baseline")\n'
        '    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), sharey=True)\n'
        '    for ax, status, color in zip(axes, ["Responder", "Nonresponder"], colors):\n'
        '        sub = pairs[pairs.response.eq(status)]\n'
        '        for _, row in sub.iterrows():\n'
        '            ax.plot([0,1], [row.TYROBP_log2_RMA_baseline, row.TYROBP_log2_RMA_followup], "o-", color=color, alpha=.65, ms=4, lw=1)\n'
        '        pv = result[f"paired_{status}_wilcoxon_p"]\n'
        '        ax.set_xticks([0,1], ["Baseline",cfg["week"]])\n'
        '        ax.set_title(f"{status} (n={len(sub)})\\nPaired P = {pv:.4g}")\n'
        '        ax.spines[["top","right"]].set_visible(False)\n'
        '    axes[0].set_ylabel("TYROBP expression (RMA log2)")\n'
        '    fig.suptitle(f"GSE73661: {cfg[\'treatment\']}")\n'
        '    fig.tight_layout()\n'
        '    save_figure(fig, f"{cfg[\'endpoint\']}_paired")\n'
        '    return result\n'
        '\n'
        '\n'
        'def main():\n'
        '    OUT.mkdir(exist_ok=True)\n'
        '    FIG.mkdir(exist_ok=True)\n'
        '    fetch_if_missing()\n'
        '    df = parse_metadata()\n'
        '    df = df.merge(read_expression(tyrobp_probe()), on="gsm", validate="one_to_one")\n'
        '    df.to_csv(OUT / "all_samples_TYROBP_metadata.csv", index=False, encoding="utf-8-sig")\n'
        '    df[df.week_title_conflict].to_csv(OUT / "metadata_week_title_conflicts.csv", index=False, encoding="utf-8-sig")\n'
        '    df.groupby(["regimen","week"],dropna=False).size().rename("n").reset_index().to_csv(OUT / "sample_counts_regimen_week.csv", index=False)\n'
        '    rng = np.random.default_rng(SEED)\n'
        '    results = [run_endpoint(df, cfg, rng) for cfg in ENDPOINTS]\n'
        '    pd.DataFrame(results).to_csv(OUT / "summary_statistics.csv", index=False, encoding="utf-8-sig")\n'
        '    (OUT / "summary_statistics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")\n'
        '    versions = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,\n'
        '                "scipy": scipy.__version__, "matplotlib": matplotlib.__version__, "scikit-learn": sklearn.__version__, "seed": SEED}\n'
        '    (OUT / "software_versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")\n'
        '    print(pd.DataFrame(results)[["endpoint","n_responder","n_nonresponder","mean_difference_NR_minus_R","p_value","auc_higher_predicts_nonresponse","auc_boot95_low","auc_boot95_high"]].to_string(index=False))\n'
        '\n'
        '\n'
        'if __name__ == "__main__":\n'
        '    main()\n'
    ),
    'fig09_build_report.py': (
        '"""Combine the three reproducible dataset analyses into a clinical-relevance report.\n'
        '\n'
        'Run from the delivered folder: python build_report.py\n'
        'The source datasets remain separate; no cross-platform expression pooling is done.\n'
        '"""\n'
        'from pathlib import Path\n'
        'import csv\n'
        'import gzip\n'
        'import hashlib\n'
        'import html\n'
        'import json\n'
        'import platform\n'
        'import sys\n'
        'import importlib.metadata\n'
        'import numpy as np\n'
        'import pandas as pd\n'
        'from scipy import stats\n'
        'from statsmodels.stats.multitest import multipletests\n'
        'import matplotlib\n'
        "matplotlib.use('Agg')\n"
        'import matplotlib.pyplot as plt\n'
        '\n'
        'ROOT = Path(__file__).resolve().parent\n'
        "DATA = ROOT / 'datasets'\n"
        "TABLE = ROOT / 'tables'\n"
        "FIG = ROOT / 'figures'\n"
        'for folder in (TABLE, FIG):\n'
        '    folder.mkdir(exist_ok=True)\n'
        '\n'
        'def ptext(p):\n'
        "    return f'{p:.3g}' if p < .001 else f'{p:.4f}'\n"
        '\n'
        'def read_activity():\n'
        '    """Independent extraction of the uniquely mapped transcript cluster 8036224.\n'
        '\n'
        '    The dedicated GSE75214 script also validates its exact TYROBP platform mapping.\n'
        '    This extraction verifies sample filtering directly from the deposited matrix.\n'
        '    """\n'
        '    meta, values, sample_ids = [], None, None\n'
        "    with gzip.open(DATA/'gse75214'/'GSE75214_series_matrix.txt.gz', 'rt', encoding='utf-8-sig') as fh:\n"
        '        for line in fh:\n'
        "            if line.startswith('!Sample_characteristics_ch1'):\n"
        "                meta.append(next(csv.reader([line], delimiter='\\t'))[1:])\n"
        "            elif line.startswith('!Sample_geo_accession'):\n"
        "                sample_ids = next(csv.reader([line], delimiter='\\t'))[1:]\n"
        "            elif line.startswith('!series_matrix_table_begin'):\n"
        "                header = next(csv.reader([next(fh)], delimiter='\\t'))[1:]\n"
        '                assert header == sample_ids\n'
        "            elif '8036224' in line[:25]:\n"
        "                row = next(csv.reader([line], delimiter='\\t'))\n"
        "                if row[0] == '8036224':\n"
        '                    values = np.asarray(row[1:], float)\n'
        '                    break\n'
        '    assert values is not None\n'
        "    d = pd.DataFrame({'gsm': sample_ids, 'TYROBP_log2': values})\n"
        '    for row in meta:\n'
        "        split = [x.partition(': ') for x in row]\n"
        '        labels = {x[0] for x in split}\n'
        '        assert len(labels) == 1\n'
        '        d[next(iter(labels))] = [x[2] for x in split]\n'
        "    d = d[d.tissue.eq('colon') & d.disease.isin(['control','ulcerative colitis'])].copy()\n"
        "    d['group'] = d['disease activity'].map({'normal':'Control','inactive':'Inactive UC','active':'Active UC'})\n"
        "    assert d.groupby('group').size().to_dict() == {'Active UC':74, 'Control':11, 'Inactive UC':23}\n"
        '    assert d.gsm.is_unique and d.TYROBP_log2.notna().all()\n'
        "    d.to_csv(TABLE/'GSE75214_analyzed_samples.csv', index=False, encoding='utf-8-sig')\n"
        "    groups = ['Control','Inactive UC','Active UC']\n"
        "    arrays = {g:d.loc[d.group.eq(g),'TYROBP_log2'].to_numpy() for g in groups}\n"
        '    omnibus = stats.kruskal(*(arrays[g] for g in groups))\n'
        '    comparisons = []\n'
        "    for low, high in [('Control','Inactive UC'),('Control','Active UC'),('Inactive UC','Active UC')]:\n"
        "        test = stats.mannwhitneyu(arrays[high], arrays[low], alternative='two-sided', method='auto')\n"
        "        comparisons.append({'group1':low,'group2':high,'n1':len(arrays[low]),'n2':len(arrays[high]),\n"
        "            'mean1':arrays[low].mean(),'mean2':arrays[high].mean(),\n"
        "            'median1':np.median(arrays[low]),'median2':np.median(arrays[high]),\n"
        "            'mean_difference_group2_minus_group1':arrays[high].mean()-arrays[low].mean(),\n"
        "            'p_raw':test.pvalue, 'AUC_higher_predicts_group2':test.statistic/(len(arrays[low])*len(arrays[high]))})\n"
        '    pairwise = pd.DataFrame(comparisons)\n'
        "    pairwise['p_Holm_3_activity_comparisons'] = multipletests(pairwise.p_raw, method='holm')[1]\n"
        "    pairwise.to_csv(TABLE/'activity_comparisons.csv', index=False, encoding='utf-8-sig')\n"
        '    return d, arrays, omnibus, pairwise\n'
        '\n'
        'def read_response():\n'
        "    one = json.loads((DATA/'gse16879'/'results_GSE16879.json').read_text(encoding='utf-8'))\n"
        "    a = one['primary_analysis']\n"
        "    rows = [{'dataset':'GSE16879','endpoint':'IFX_W4_W6','treatment':'Infliximab','n_R':a['n_responders'],\n"
        "        'n_NR':a['n_nonresponders'],'mean_R':a['mean_responders_log2'],'mean_NR':a['mean_nonresponders_log2'],\n"
        "        'p_raw':a['p_value'],'AUC_high_predicts_nonresponse':a['auc_fixed_direction_high_expression_predicts_nonresponse'],\n"
        "        'AUC_CI_low':a['auc_stratified_bootstrap_95CI'][0], 'AUC_CI_high':a['auc_stratified_bootstrap_95CI'][1],\n"
        "        'family':'primary_baseline'}]\n"
        "    multiple = json.loads((DATA/'gse73661'/'results'/'summary_statistics.json').read_text(encoding='utf-8'))\n"
        '    for b in multiple:\n'
        "        rows.append({'dataset':'GSE73661','endpoint':b['endpoint'],'treatment':b['treatment'],\n"
        "            'n_R':b['n_responder'],'n_NR':b['n_nonresponder'],'mean_R':b['baseline_responder_mean'],\n"
        "            'mean_NR':b['baseline_nonresponder_mean'],'p_raw':b['p_value'],\n"
        "            'AUC_high_predicts_nonresponse':b['auc_higher_predicts_nonresponse'],\n"
        "            'AUC_CI_low':b['auc_boot95_low'],'AUC_CI_high':b['auc_boot95_high'],\n"
        "            'family':'secondary_baseline' if b['endpoint']=='VDZ_W52' else 'primary_baseline'})\n"
        '    d = pd.DataFrame(rows)\n'
        "    main = d.family.eq('primary_baseline')\n"
        '    assert main.sum()==4\n'
        "    d.loc[main,'p_Holm_4_primary_baseline_comparisons'] = multipletests(d.loc[main,'p_raw'],method='holm')[1]\n"
        "    d['mean_difference_NR_minus_R'] = d.mean_NR-d.mean_R\n"
        "    d.to_csv(TABLE/'baseline_response_comparisons.csv', index=False, encoding='utf-8-sig')\n"
        '    long = []\n'
        "    for row in one['secondary_longitudinal']:\n"
        "        long.append({'dataset':'GSE16879','endpoint':'IFX_W4_W6','analysis':row['analysis'],'p_raw':row['p_value']})\n"
        '    for b in multiple:\n'
        "        for name, key in [('Responder: paired post-treatment vs baseline','paired_Responder_wilcoxon_p'),\n"
        "                          ('Nonresponder: paired post-treatment vs baseline','paired_Nonresponder_wilcoxon_p'),\n"
        "                          ('Post-minus-baseline change: nonresponders vs responders','change_difference_MW_p')]:\n"
        "            long.append({'dataset':'GSE73661','endpoint':b['endpoint'],'analysis':name,'p_raw':b[key]})\n"
        '    longitudinal = pd.DataFrame(long)\n'
        "    longitudinal['p_Holm_15_secondary_longitudinal_comparisons'] = multipletests(longitudinal.p_raw,method='holm')[1]\n"
        "    longitudinal.to_csv(TABLE/'longitudinal_comparisons.csv', index=False, encoding='utf-8-sig')\n"
        '    return d, longitudinal\n'
        '\n'
        'def figures(activity, arrays, omnibus, pairwise, responses):\n'
        "    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})\n"
        '    rng = np.random.default_rng(20261115)\n'
        '    fig,ax=plt.subplots(figsize=(7.2,5))\n'
        "    colors=['#3B7599','#D4A34A','#B54A62']\n"
        '    groups=list(arrays)\n'
        '    box=ax.boxplot([arrays[g] for g in groups],positions=range(3),widths=.5,showfliers=False,patch_artist=True)\n'
        "    for i,(g,color,patch) in enumerate(zip(groups,colors,box['boxes'])):\n"
        '        patch.set(facecolor=color,alpha=.22)\n'
        "        ax.scatter(i+rng.uniform(-.13,.13,len(arrays[g])),arrays[g],s=24,color=color,edgecolor='white',linewidth=.35,zorder=3)\n"
        '    upper=activity.TYROBP_log2.max()\n'
        '    for step,(i,j,ix) in enumerate([(0,1,0),(1,2,2),(0,2,1)]):\n'
        '        y=upper+.16+step*.27\n'
        "        ax.plot([i,i,j,j],[y-.04,y,y,y-.04],c='.25',lw=.8)\n"
        '        p=pairwise.iloc[ix].p_Holm_3_activity_comparisons\n'
        "        ax.text((i+j)/2,y+.025,f'Holm P = {ptext(p)}',ha='center',va='bottom',fontsize=9)\n"
        "    ax.set_xticks(range(3),[f'{g}\\n(n = {len(arrays[g])})' for g in groups])\n"
        "    ax.set_ylabel('TYROBP expression (RMA log$_2$ intensity)')\n"
        "    ax.set_title(f'GSE75214: colonic disease activity\\nKruskal-Wallis P = {ptext(omnibus.pvalue)}',pad=12)\n"
        '    ax.set_ylim(top=upper+1.12)\n'
        '    fig.tight_layout()\n'
        "    for ext in ['png','pdf','svg']: fig.savefig(FIG/f'01_TYROBP_disease_activity.{ext}',dpi=300,bbox_inches='tight')\n"
        '    plt.close(fig)\n'
        '    fig,axes=plt.subplots(2,2,figsize=(10,8))\n'
        "    for ax,(_,row) in zip(axes.ravel(),responses[responses.family.eq('primary_baseline')].iterrows()):\n"
        "        if row.dataset=='GSE16879':\n"
        "            src=pd.read_csv(DATA/'gse16879'/'UC_baseline_TYROBP.csv')\n"
        "            values=[src.loc[src.response_group.eq(g),'TYROBP_log2'].to_numpy() for g in ['Responder','Nonresponder']]\n"
        '        else:\n'
        "            src=pd.read_csv(DATA/'gse73661'/'results'/f'{row.endpoint}_paired_samples.csv')\n"
        "            values=[src.loc[src.response.eq(g),'TYROBP_log2_RMA_baseline'].to_numpy() for g in ['Responder','Nonresponder']]\n"
        '        assert [len(x) for x in values]==[row.n_R,row.n_NR]\n'
        "        test=stats.mannwhitneyu(values[1],values[0],alternative='two-sided',method='auto')\n"
        '        assert abs(test.pvalue-row.p_raw)<1e-10\n'
        '        bp=ax.boxplot(values,positions=[0,1],widths=.48,showfliers=False,patch_artist=True)\n'
        "        for i,(v,c,patch) in enumerate(zip(values,['#3B7599','#B54A62'],bp['boxes'])):\n"
        '            patch.set(facecolor=c,alpha=.22)\n'
        "            ax.scatter(i+rng.uniform(-.12,.12,len(v)),v,s=28,color=c,edgecolor='white',linewidth=.4,zorder=3)\n"
        "        ax.set_xticks([0,1],[f'Responders\\n(n = {int(row.n_R)})',f'Nonresponders\\n(n = {int(row.n_NR)})'])\n"
        "        endpoint=row.endpoint.replace('IFX_','').replace('VDZ_','').replace('W4_W6','Weeks 4-6').replace('W6','Week 6').replace('W12','Week 12')\n"
        "        ax.set_title(f'{row.dataset}: {row.treatment}, {endpoint}\\nP = {ptext(row.p_raw)}; Holm P = {ptext(row.p_Holm_4_primary_baseline_comparisons)}',fontsize=10)\n"
        "        ax.set_ylabel('Baseline TYROBP (log$_2$ intensity)')\n"
        "    fig.suptitle('Baseline expression and subsequent biologic response',fontsize=14,y=1.01)\n"
        '    fig.tight_layout()\n'
        "    for ext in ['png','pdf','svg']: fig.savefig(FIG/f'02_TYROBP_baseline_response.{ext}',dpi=300,bbox_inches='tight')\n"
        '    plt.close(fig)\n'
        '    fig,ax=plt.subplots(figsize=(8,4.7))\n'
        '    for i,(_,r) in enumerate(responses.iterrows()):\n'
        "        color='#9B9B9B' if r.family=='secondary_baseline' else '#3B7599'\n"
        "        ax.errorbar(r.AUC_high_predicts_nonresponse,i,xerr=[[r.AUC_high_predicts_nonresponse-r.AUC_CI_low],[r.AUC_CI_high-r.AUC_high_predicts_nonresponse]],fmt='o',color=color,capsize=4)\n"
        "    ax.axvline(.5,color='.5',ls='--',lw=1)\n"
        '    ax.set_yticks(range(len(responses)),[f\'{r.dataset} | {r.treatment} | {r.endpoint.split("_",1)[1].replace("_","-")}\' + (\' (secondary)\' if r.family==\'secondary_baseline\' else \'\') for _,r in responses.iterrows()])\n'
        '    ax.invert_yaxis(); ax.set_xlim(0,1)\n'
        "    ax.set_xlabel('AUC (higher baseline TYROBP predicts nonresponse)')\n"
        "    ax.set_title('Exploratory discrimination: 95% stratified-bootstrap intervals\\nNo direction reversal or optimized cutoff; no pooled estimate',fontsize=11)\n"
        '    fig.tight_layout()\n'
        "    for ext in ['png','pdf','svg']: fig.savefig(FIG/f'03_TYROBP_response_AUC.{ext}',dpi=300,bbox_inches='tight')\n"
        '    plt.close(fig)\n'
        '\n'
        'def report(activity,arrays,omnibus,pairwise,responses,longitudinal):\n'
        "    act=pairwise.set_index(['group1','group2'])\n"
        "    pa=float(act.loc[('Inactive UC','Active UC'),'p_Holm_3_activity_comparisons'])\n"
        "    pc=float(act.loc[('Control','Active UC'),'p_Holm_3_activity_comparisons'])\n"
        "    pi=float(act.loc[('Control','Inactive UC'),'p_Holm_3_activity_comparisons'])\n"
        '    tables=[]\n'
        "    ar=pairwise[['group1','group2','n1','n2','mean_difference_group2_minus_group1','p_raw','p_Holm_3_activity_comparisons','AUC_higher_predicts_group2']].copy()\n"
        "    rr=responses[['dataset','endpoint','n_R','n_NR','mean_difference_NR_minus_R','p_raw','p_Holm_4_primary_baseline_comparisons','AUC_high_predicts_nonresponse','AUC_CI_low','AUC_CI_high']].copy()\n"
        '    for table in [ar,rr,longitudinal]:\n'
        "        tables.append(table.to_html(index=False,classes='data',border=0,float_format=lambda x:f'{x:.5g}',na_rep='Not in primary family'))\n"
        "    headline=(f'GSE75214：活动期 UC 高于非活动期 UC（Holm P = {ptext(pa)}）及对照（Holm P = {ptext(pc)}）；'\n"
        "              f'非活动期 UC 与对照无显著差异（P = {ptext(pi)}）。四项主要基线治疗应答比较均未显著，所有 AUC 的 95% 区间均跨越 0.5。')\n"
        "    methods_en=('Official GEO processed expression matrices and platform annotations were used. GSE75214 was restricted to colonic biopsies from controls (n=11), inactive UC (n=23), and active UC (n=74), using the source-defined endoscopic activity groups. TYROBP mapped uniquely to transcript cluster 8036224 on GPL6244 and probe 204122_at on GPL570. RMA log2 values in GSE75214/GSE73661 were retained without further log transformation; GSE16879 GCOS globally scaled positive signals were log2-transformed. Activity groups were compared using Kruskal-Wallis and two-sided Mann-Whitney tests, with Holm adjustment across three pairwise contrasts. Baseline treatment-response analyses were restricted to UC and performed separately for each dataset, drug, and assessment window. GSE73661 response was defined by follow-up Mayo endoscopic subscore <=1; GSE16879 used the deposited endoscopic/histologic response labels. Holm adjustment covered four primary baseline comparisons: GSE16879 IFX weeks 4-6, GSE73661 IFX weeks 4-6, VDZ week 6, and the separate VDZ week-12 cohort. VDZ week 52 and longitudinal paired analyses were secondary. Higher baseline TYROBP was fixed as the direction for predicting nonresponse in exploratory ROC analyses; 95% AUC intervals used 10,000 stratified bootstrap draws. No cutoff optimization or expression pooling across datasets was performed.')\n"
        "    result_en=(f'In GSE75214, TYROBP expression differed among activity groups (Kruskal-Wallis P = {ptext(omnibus.pvalue)}). Expression was higher in active UC than in inactive UC (Holm-adjusted P = {ptext(pa)}) and controls (Holm-adjusted P = {ptext(pc)}), whereas inactive UC did not differ significantly from controls (Holm-adjusted P = {ptext(pi)}). Baseline expression did not differ significantly between response groups for infliximab in GSE16879 (8 responders/16 nonresponders; P = 0.1531) or GSE73661 (8/15; P = 0.2939), or for vedolizumab at week 6 (6/21; P = 0.1569) or week 12 (3/10; P = 0.5734). All four Holm-adjusted P values were 0.6125. Their fixed-direction AUCs were 0.6875, 0.6417, 0.6984 and 0.6333, respectively, and all 95% intervals included 0.5. The secondary continuous-VDZ week-52 comparison was also nonsignificant (12/5; P = 0.1296). These results support an association with inflammatory disease activity but do not establish a clinically validated diagnostic or treatment-response biomarker.')\n"
        "    reply_zh=(f'感谢审稿人的建议。我们补充分析了 GSE75214 中的结肠活检样本，包括 11 例对照、23 例非活动期 UC 和 74 例活动期 UC。TYROBP 在活动期 UC 中的表达高于非活动期 UC（Holm 校正 P = {ptext(pa)}）和对照组（Holm 校正 P = {ptext(pc)}），而非活动期 UC 与对照组无显著差异（P = {ptext(pi)}）。我们还分别分析了 GSE16879 和 GSE73661 中治疗前的 UC 样本，按原研究定义和评估时间比较后续生物制剂应答者与非应答者。两项英夫利昔单抗比较及维多珠单抗第 6 周、第 12 周比较均未达到统计学显著；四项比较的 Holm 校正 P 均为 0.6125。长期维多珠单抗第 52 周比较也未显著。上述结果提示 TYROBP 更适合被定位为与 UC 炎症活动相关的髓系/巨噬细胞表达信号，现有证据不足以支持其作为经临床验证的诊断或疗效预测标志物。我们据此拟收敛相关结论，并在修订稿中报告这些阳性和阴性结果。')\n"
        "    reply_en=(f'We thank the reviewer for this suggestion. We additionally analyzed colonic biopsies from GSE75214, including 11 controls, 23 patients with inactive UC, and 74 with active UC. TYROBP expression was higher in active UC than in inactive UC (Holm-adjusted P = {ptext(pa)}) and controls (Holm-adjusted P = {ptext(pc)}), whereas inactive UC did not differ significantly from controls (P = {ptext(pi)}). We also examined pretreatment UC samples in GSE16879 and GSE73661, comparing subsequent biologic responders and nonresponders using the original study definitions and assessment windows. Neither infliximab comparison nor the vedolizumab week-6 or week-12 comparison reached statistical significance; all four Holm-adjusted P values were 0.6125. The secondary week-52 vedolizumab comparison was also nonsignificant. These findings support positioning TYROBP as a myeloid/macrophage-associated expression signal related to UC inflammatory activity. The present evidence does not establish a clinically validated diagnostic or treatment-response biomarker. We therefore propose narrowing the relevant claims and reporting both the positive and negative findings in the revised manuscript.')\n"
        "    plain='TYROBP 临床相关性分析 / Clinical relevance analysis\\n2026-11-15\\n\\n'+headline+'\\n\\nMETHODS (English draft)\\n'+methods_en+'\\n\\nRESULTS (English draft)\\n'+result_en+'\\n\\n审稿人回复草稿（中文；尚需同步修改论文）\\n'+reply_zh+'\\n\\nReviewer response draft (English; manuscript changes remain to be made)\\n'+reply_en+'\\n'\n"
        "    (ROOT/'Results_and_reviewer_response_ZH_EN.txt').write_text(plain,encoding='utf-8-sig')\n"
        "    limitations=['非活动期按原数据的内镜活动度标签定义，不等同于已证实的组织学完全缓解。',\n"
        "        '主要比较为未调整协变量的单基因探索性分析；组织细胞组成及炎症程度可能影响表达。',\n"
        "        '治疗应答组样本量较小，特别是 VDZ 第12周应答者仅3例；未显著并不证明完全没有关联。',\n"
        "        'ROC 使用预先固定方向，未优化阈值；区间为分层 bootstrap 百分位区间，不是前瞻性临床验证。',\n"
        "        'GSE73661 中基线到终点缺失的患者不填补应答；第52周仅分析有随访的持续VDZ治疗者，存在选择/失访偏倚。',\n"
        "        'GSE73661 的 vdz_plac 分支可用于第6周诱导终点，第52周因安慰剂维持而排除；placebo-only 分支排除。',\n"
        "        '三个 GSE73661 样本标题的周数与结构化字段冲突，使用结构化 week 字段，并保留冲突清单。',\n"
        "        '两项 IFX 数据集的患者独立性未获原始资料证实，分别分析，不合并，也不称为独立患者重复验证。',\n"
        "        'GSE16879 应答同时参考内镜与组织学；GSE73661 用内镜 Mayo <=1。不同终点不合并。',\n"
        "        '均值差的 bootstrap 区间描述均值估计，Mann-Whitney P 检验秩分布；两者的统计目标不同。',\n"
        "        '15项纵向检验作为次要分析另行 Holm 校正；“某组显著、另一组不显著”不代表两组变化幅度有差异。',\n"
        "        '原始 CEL 文件未重新处理，保留 GEO 提交者的预处理；本分析不确认 TYROBP 的细胞特异性或因果作用。']\n"
        "    sources=[('GSE75214','https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE75214'),('GSE73661','https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE73661'),('GSE16879','https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE16879'),('GPL6244','https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL6244'),('GPL570','https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL570')]\n"
        '    content=f\'\'\'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>TYROBP 临床相关性分析</title>\n'
        '    <style>body{{font-family:"Segoe UI","Microsoft YaHei",sans-serif;color:#183044;background:#f3f6f8;margin:0;line-height:1.7}}main{{max-width:1180px;margin:30px auto;background:white;padding:34px 44px}}h1{{font-size:27px;margin-bottom:4px}}h2{{font-size:20px;margin-top:34px;border-bottom:1px solid #dae1e6;padding-bottom:6px}}.date,.note{{color:#536b7b}}.lead{{background:#edf5f7;border-left:4px solid #367b95;padding:18px}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;font-size:13px;width:100%}}th,td{{padding:8px;border-bottom:1px solid #dbe3e8;text-align:left}}th{{background:#edf2f5}}img{{max-width:100%;height:auto}}p{{text-align:justify}}a{{color:#176987}}li{{margin:7px 0}}@media print{{body{{background:white}}main{{margin:0;padding:0}}h2{{break-after:avoid}}img{{max-height:23cm;object-fit:contain}}}}</style>\n'
        '    <main><h1>TYROBP 临床相关性分析</h1><div class="date">2026-11-15 · 公开结肠组织数据 · 探索性单基因分析</div>\n'
        '    <p class="lead">{headline}</p>\n'
        '    <h2>1. 疾病活动度：对照、非活动期 UC、活动期 UC</h2>\n'
        "    <p>严格限定结肠组织：11例对照、23例非活动期UC、74例活动期UC。三组均值依次为 {arrays['Control'].mean():.3f}、{arrays['Inactive UC'].mean():.3f}、{arrays['Active UC'].mean():.3f}（RMA log2）。总体 Kruskal-Wallis P = {ptext(omnibus.pvalue)}。下表差值方向为 group2 − group1。</p>\n"
        '    <img src="figures/01_TYROBP_disease_activity.png" alt="TYROBP expression by UC activity"><div class="scroll">{tables[0]}</div>\n'
        '    <p><a href="tables/activity_comparisons.csv">下载活动度统计表</a> · <a href="tables/GSE75214_analyzed_samples.csv">下载纳入样本及表达</a></p>\n'
        '    <h2>2. 治疗前表达与后续生物制剂应答</h2>\n'
        '    <p>IFX为英夫利昔单抗，VDZ为维多珠单抗。按数据集、药物和随访时间分别比较，未将治疗后表达当作治疗前预测。四项主要比较采用Holm校正，第52周列为次要探索。表中差值为非应答者 − 应答者。AUC固定高表达指向非应答，未自动反转方向；所有AUC区间均包含0.5。</p>\n'
        '    <img src="figures/02_TYROBP_baseline_response.png" alt="Baseline response comparisons"><div class="scroll">{tables[1]}</div>\n'
        '    <img src="figures/03_TYROBP_response_AUC.png" alt="AUC intervals"><p><a href="tables/baseline_response_comparisons.csv">下载全部基线比较结果</a></p>\n'
        '    <h2>3. 治疗前后的配对变化（次要分析）</h2>\n'
        '    <p>按明确患者编号配对。部分未校正组内P值较小，但应答者与非应答者的表达变化幅度比较均未显著。15项次要纵向检验经统一Holm校正后均未显著；不能把治疗后变化解释为已证实的基线预测能力。</p><div class="scroll">{tables[2]}</div>\n'
        '    <p>各终点完整的配对轨迹图及患者表位于 datasets/gse16879 和 datasets/gse73661/figures。</p>\n'
        '    <h2>4. 论文定位与可用文字</h2><p>更稳妥的定位是“与UC炎症活动相关的髓系/巨噬细胞表达信号”。本分析没有显示其能区分非活动期UC与健康对照，也未取得明确的基线疗效预测证据。现有资料不足以支持已验证的临床诊断或疗效预测用途。</p>\n'
        '    <h3>Methods draft</h3><p>{html.escape(methods_en)}</p><h3>Results draft</h3><p>{html.escape(result_en)}</p>\n'
        '    <h3>审稿人回复草稿（中文）</h3><p>{html.escape(reply_zh)}</p><h3>Reviewer response draft (English)</h3><p>{html.escape(reply_en)}</p>\n'
        '    <p class="note">以上为基于本次分析的写作草稿，未自动修改原论文。提交前须同步更新正文、摘要、图表及诊断性措辞。</p>\n'
        "    <h2>5. 方法边界与复现</h2><ul>{''.join('<li>'+html.escape(x)+'</li>' for x in limitations)}</ul>\n"
        '    <p>数据来自 GEO 提交者处理矩阵，探针经官方平台注释核对。原始下载文件、样本筛选、患者配对、精确统计量、软件版本和分析脚本均随包提供。运行：<code>python run_all.py</code>。单独重建本报告：<code>python build_report.py</code>。</p>\n'
        '    <p>主要数据源：{ \' · \'.join(f\'<a href="{url}">{name}</a>\' for name,url in sources)}</p>\n'
        '    <p><a href="Results_and_reviewer_response_ZH_EN.txt">下载中英文结果与回复草稿</a> · <a href="README.txt">查看运行说明</a></p></main></html>\'\'\'\n'
        "    (ROOT/'TYROBP_clinical_report.html').write_text(content,encoding='utf-8')\n"
        "    summary={'activity_omnibus_p':float(omnibus.pvalue),'activity_comparisons':pairwise.to_dict('records'),\n"
        "        'baseline_response_comparisons':responses.replace({np.nan:None}).to_dict('records'),\n"
        "        'longitudinal_comparisons':longitudinal.to_dict('records'),'headline_zh':headline}\n"
        "    (TABLE/'consolidated_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')\n"
        '\n'
        'def manifest():\n'
        '    rows=[]\n'
        "    for f in sorted(ROOT.rglob('*')):\n"
        "        if f.is_file() and f.name not in ['file_manifest_sha256.csv'] and '__pycache__' not in f.parts:\n"
        "            rows.append({'relative_path':f.relative_to(ROOT).as_posix(),'bytes':f.stat().st_size,\n"
        "                         'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})\n"
        "    pd.DataFrame(rows).to_csv(ROOT/'file_manifest_sha256.csv',index=False,encoding='utf-8-sig')\n"
        '\n'
        "if __name__=='__main__':\n"
        '    activity,arrays,omnibus,pairwise=read_activity()\n'
        '    responses,longitudinal=read_response()\n'
        '    assert (responses.AUC_CI_low<.5).all() and (responses.AUC_CI_high>.5).all()\n'
        '    assert (responses.p_raw>.05).all()\n'
        '    assert (longitudinal.p_Holm_15_secondary_longitudinal_comparisons>.05).all()\n'
        '    figures(activity,arrays,omnibus,pairwise,responses)\n'
        '    report(activity,arrays,omnibus,pairwise,responses,longitudinal)\n'
        '    from beautify_figures import main as redraw_reference_style\n'
        '    redraw_reference_style()\n'
        '    manifest()\n'
        "    print(json.dumps({'report':str(ROOT/'TYROBP_clinical_report.html'),'activity_p':omnibus.pvalue,'response_comparisons':len(responses)},ensure_ascii=False))\n"
    ),
}

if __name__ == "__main__":
    python_cli()
