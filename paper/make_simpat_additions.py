"""Additions for the simulation-methodology version of the manuscript (no new experiments needed).

1. generated/simpat_table_paired.tex: statistical efficiency of the paired protocol (common random numbers),
   from the per-episode coverages in mate/results/paper/{main,published}_4v8.json.
2. figures/v2_burst.pdf: coverage under bursty and non-uniform loss, every point drawn at the mean loss rate
   that the link model really produces (the nominal rate is not reachable in three settings, see below).
3. generated/simpat_macros.tex: the numbers quoted in the text.

It does not touch the other generated files (their bold / underline marks were set by hand).

    python3 paper-jnca/make_simpat_additions.py
"""

import json
import os
import re

import numpy as np
from scipy import stats

from make_figures_v2 import FIGURES, GENERATED, SHORT, TARGET_TITLES, TARGETS, load, plt, rows_by, style


MACROS = {}


def macro(name, value, digits=3):
    MACROS[name] = value if isinstance(value, str) else f'{value:.{digits}f}'


# 1. Paired comparisons ---------------------------------------------------------------------------- #

COMPARISONS = [  # label, (file, variant) of the first method, (file, variant) of the second method
    ('\\FS{} vs.\\ rule-based cameras', ('main', 'traj-2-forecast'), ('main', 'greedy')),
    ('\\FS{} vs.\\ option-level planning', ('main', 'traj-2-forecast'), ('main', 'foresight')),
    ('No communication vs.\\ 2 rounds', ('main', 'traj-2-none'), ('main', 'traj-2')),
    ('2 rounds vs.\\ 1 round', ('main', 'traj-2'), ('main', 'traj-1')),
    ('3 rounds vs.\\ 2 rounds', ('main', 'traj-3'), ('main', 'traj-2')),
    ('Learned vs.\\ analytic forecasts', ('main', 'traj-2-forecast'), ('main', 'traj-2')),
    ('Published every 4th step vs.\\ 2 rounds', ('published', 'pub-period-4'), ('main', 'traj-2')),
]


def holm(pvalues):
    """Holm step-down adjusted p-values."""

    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (p.size - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def pvalue(p):
    return '$<$0.001' if p < 0.001 else f'{p:.3f}'


def paired_table():
    sources = {name: rows_by(load(f'paper/{name}_4v8.json')) for name in ('main', 'published')}
    stats_rows = []
    for targets in TARGETS:
        for label, (fa, va), (fb, vb) in COMPARISONS:
            a = np.asarray(sources[fa][(targets, va)]['per_episode_coverage'])
            b = np.asarray(sources[fb][(targets, vb)]['per_episode_coverage'])
            n, d = a.size, a - b
            se_paired = d.std(ddof=1) / np.sqrt(n)
            se_independent = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / n)
            stats_rows.append({
                'targets': targets, 'label': label, 'n': n, 'delta': d.mean(), 'se_paired': se_paired,
                'se_independent': se_independent, 'reduction': (se_independent / se_paired) ** 2,
                'correlation': np.corrcoef(a, b)[0, 1],
                'p_paired': stats.ttest_rel(a, b).pvalue, 'p_wilcoxon': stats.wilcoxon(a, b).pvalue,
                'p_independent': stats.ttest_ind(a, b, equal_var=False).pvalue,
            })
    for key in ('p_paired', 'p_independent', 'p_wilcoxon'):  # one family: all comparisons, both target models
        for row, adjusted in zip(stats_rows, holm([r[key] for r in stats_rows])):
            row[key + '_holm'] = adjusted
    lines = []
    for targets in TARGETS:
        lines.append(f'\\multicolumn{{7}}{{l}}{{\\emph{{{TARGET_TITLES[targets]}}}}} \\\\')
        for row in (r for r in stats_rows if r['targets'] == targets):
            lines.append(' & '.join([
                '\\quad ' + row['label'], f'{row["delta"]:+.3f}', f'{row["se_paired"]:.4f}',
                f'{row["se_independent"]:.4f}', f'{row["reduction"]:.1f}', pvalue(row['p_paired_holm']),
                pvalue(row['p_independent_holm']),
            ]) + ' \\\\')
        if targets != TARGETS[-1]:
            lines.append('\\midrule')
    with open(os.path.join(GENERATED, 'simpat_table_paired.tex'), 'w', encoding='UTF-8') as file:
        file.write('\n'.join(lines) + '\n')

    reductions = np.array([r['reduction'] for r in stats_rows])
    planners = np.array([r['reduction'] for r in stats_rows if 'rule-based' not in r['label'] and 'No comm' not in r['label']])
    macro('PairedEpisodes', str(stats_rows[0]['n']))
    macro('PairedReductionMin', reductions.min(), 1)
    macro('PairedReductionMax', reductions.max(), 1)
    macro('PairedReductionPlannerMin', planners.min(), 1)
    macro('PairedReductionPlannerMedian', float(np.median(planners)), 1)
    others = np.array([r['reduction'] for r in stats_rows if 'rule-based' in r['label'] or 'No comm' in r['label']])
    macro('PairedReductionOtherMax', others.max(), 1)
    macro('PairedTests', str(len(stats_rows)))
    rounds = [r for r in stats_rows if r['label'].startswith('3 rounds')]
    macro('PairedRoundsEpisodesMin', str(int(round(min(r['reduction'] * r['n'] for r in rounds), -1))))
    macro('PairedRoundsEpisodesMax', str(int(round(max(r['reduction'] * r['n'] for r in rounds), -1))))
    for row in (r for r in stats_rows if 'rule-based' in r['label']):
        macro(f'GainFsRule{SHORT[row["targets"]]}', row['delta'])
    for row in (r for r in stats_rows if r['label'].startswith('Learned')):
        tag = SHORT[row['targets']]
        macro(f'PairedForecastP{tag}', pvalue(row['p_paired_holm']))
        macro(f'PairedForecastRawP{tag}', pvalue(row['p_paired']))
        macro(f'PairedForecastWilcoxonP{tag}', pvalue(row['p_wilcoxon_holm']))
    # where the paired t-test and the Wilcoxon signed-rank test decide differently at 0.05 (stated in the text)
    for row in stats_rows:
        if (row['p_paired_holm'] < 0.05) != (row['p_wilcoxon_holm'] < 0.05):
            print(f'TESTS DISAGREE: {row["targets"]} {row["label"]}: paired t {row["p_paired_holm"]:.4f}, '
                  f'Wilcoxon {row["p_wilcoxon_holm"]:.4f}')
    for row in stats_rows:
        print(f'{row["targets"]:9s} {row["label"]:45s} d={row["delta"]:+.4f} se_p={row["se_paired"]:.4f} '
              f'se_i={row["se_independent"]:.4f} VR={row["reduction"]:5.1f} rho={row["correlation"]:.2f} '
              f'p_paired={row["p_paired_holm"]:.2g} p_wilcoxon={row["p_wilcoxon_holm"]:.2g} '
              f'p_indep={row["p_independent_holm"]:.2g}')


# 2. Loss figure at the mean loss rate that the link model really produces --------------------------- #

NUM_NODES = 4
SPREAD = 0.8


def effective_loss(kind, p, burst=None):
    """Stationary mean loss rate of the simulated link model (mate_jax/foresight/planner.py, ``_link``).

    Bursty: Gilbert-Elliott chain with bad-to-good probability 1/B and good-to-bad probability p / (B (1 - p)),
    which exceeds 1 when p > B / (B + 1); the simulator then leaves the good state after one step.
    Non-uniform: p_i = p (1 + s (2 i / (N - 1) - 1)), clipped to [0, 1].
    """

    if kind == 'burst':
        fail = min(p / (burst * (1.0 - p)), 1.0)
        return fail / (fail + 1.0 / burst)
    if kind == 'spread':
        rates = p * (1.0 + SPREAD * (2.0 * np.arange(NUM_NODES) / (NUM_NODES - 1) - 1.0))
        return float(np.clip(rates, 0.0, 1.0).mean())
    return p


def burst_figure():
    rows = rows_by(load('paper/burst_4v8.json'))
    rows.update(rows_by(load('paper/loss_4v8.json')))  # i.i.d. loss with the shared memory and for option-level planning
    main = rows_by(load('paper/main_4v8.json'))
    losses = (0.25, 0.5, 0.75, 0.9)
    series = [('iid-{p}', 'FORESIGHT, hybrid, i.i.d.', '#1f6fb2', '-', ('iid', None), 'traj-2-hybrid'),
              ('traj-team-{p}', 'FORESIGHT, shared, i.i.d.', '#1f6fb2', '--', ('iid', None), 'traj-2'),
              ('burst5-{p}', 'FORESIGHT, hybrid, bursts of 5', '#e08b2f', '-', ('burst', 5), 'traj-2-hybrid'),
              ('burst20-{p}', 'FORESIGHT, hybrid, bursts of 20', '#b22222', '-', ('burst', 20), 'traj-2-hybrid'),
              ('spread-{p}', 'FORESIGHT, hybrid, non-uniform ($\\pm$80%)', '#2a9d8f', '--', ('spread', None), 'traj-2-hybrid'),
              ('pub-burst20-{p}', 'published intents, bursts of 20, intents lost too', '#6a3d9a', ':', ('burst', 20), None),
              ('foresight-hybrid-{p}', 'option-level, hybrid, i.i.d.', '#555555', '-', ('iid', None), 'hybrid'),
              ('foresight-team-{p}', 'option-level, shared, i.i.d.', '#999999', '--', ('iid', None), 'foresight')]
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for ax, targets in zip(axes, TARGETS):
        zero = rows[(targets, 'traj-2-hybrid')]['coverage']
        for pattern, label, color, ls, (kind, burst), start in series:
            xs = [effective_loss(kind, p, burst) for p in losses]
            ys = [rows[(targets, pattern.format(p=p))]['coverage'] for p in losses]
            if start is not None:
                xs, ys = [0.0] + xs, [main[(targets, start)]['coverage']] + ys
            order = np.argsort(xs)
            xs, ys = np.asarray(xs)[order], np.asarray(ys)[order]
            ax.plot(xs, ys, marker='o', ms=3, lw=1.4, color=color, ls=ls, label=label)
        ax.axhline(main[(targets, 'traj-2-none')]['coverage'], color='gray', ls=':', lw=0.9, label='FORESIGHT, no communication')
        ax.set_title(TARGET_TITLES[targets])
        ax.set_xlabel('mean packet loss rate')
        tag = SHORT[targets]
        for key, name in (('iid-0.75', 'LossIidHigh'), ('burst5-0.75', 'LossFiveHigh'), ('burst20-0.75', 'LossTwentyHigh'),
                          ('iid-0.5', 'LossIidHalf'), ('spread-0.5', 'LossSpreadHalf'),
                          ('spread-0.75', 'LossSpreadHigh'), ('spread-0.9', 'LossSpreadTop')):
            macro(f'{name}{tag}', rows[(targets, key)]['coverage'])
        # largest difference to i.i.d. loss of the same mean rate, over the settings where the nominal rate is reached
        valid = [(f'burst{b}-{p}', f'iid-{p}') for b in (5, 20) for p in losses
                 if abs(effective_loss('burst', p, b) - p) < 1e-9]
        gap = max(abs(rows[(targets, a)]['coverage'] - rows[(targets, b)]['coverage']) for a, b in valid)
        macro(f'LossBurstMaxDiff{tag}', gap)
        valid = [(f'spread-{p}', f'iid-{p}') for p in losses if abs(effective_loss('spread', p) - p) < 1e-9]
        gap = max(abs(rows[(targets, a)]['coverage'] - rows[(targets, b)]['coverage']) for a, b in valid)
        macro(f'LossSpreadMaxDiff{tag}', gap)
        # i.i.d. coverage interpolated at the mean rate of the two saturated non-uniform settings
        iid_x = [0.0] + list(losses)
        iid_y = [zero] + [rows[(targets, f'iid-{p}')]['coverage'] for p in losses]
        for p, name in ((0.75, 'High'), (0.9, 'Top')):
            reference = float(np.interp(effective_loss('spread', p), iid_x, iid_y))
            macro(f'LossSpread{name}Drop{tag}', reference - rows[(targets, f'spread-{p}')]['coverage'])
    axes[0].set_ylabel('coverage rate')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.22, 1, 1))
    fig.savefig(os.path.join(FIGURES, 'simpat_loss.pdf'))
    plt.close(fig)
    macro('LossFiveTopRate', effective_loss('burst', 0.9, 5), 2)
    macro('LossSpreadHighRate', effective_loss('spread', 0.75), 2)
    macro('LossSpreadTopRate', effective_loss('spread', 0.9), 2)


# 3. Multi-step parity of MATE-JAX with the original MATE --------------------------------------------- #

def sci(value):
    """Small numbers as 10^k in LaTeX."""

    if value == 0.0:
        return '0'
    if value >= 0.1:
        return f'{value:.0f}'
    exponent = int(np.floor(np.log10(value)))
    return f'${value / 10.0 ** exponent:.0f}\\times10^{{{exponent}}}$'


def parity_table():
    data = load('paper/parity_multistep.json')
    if not data:
        print('parity: no results yet')
        return
    rows = {(r['config'], r['transmittance'], r['precision'], r['steps']): r for r in data['trace_replay']}
    configs = ('MATE-4v8-9.yaml', 'MATE-8v8-9.yaml')
    lines = []
    for config in configs:  # deterministic visibility (transmittance 0): any difference is an implementation difference
        for precision in ('float64', 'float32'):
            last = rows[(config, 0.0, precision, 1000)]
            lines.append(' & '.join(
                [config.replace('MATE-', '').replace('-9.yaml', ''), precision.replace('float', '') + ' bit']
                + [sci(rows[(config, 0.0, precision, k)]['max_position_error']) for k in (1, 10, 100, 1000)]
                + [f'{100.0 * last["diverged_target_share"]:.1f}', sci(last['max_orientation_error']),
                   f'{100.0 * last["visibility_mismatch_rate"]:.3f}',
                   f'{last["coverage_difference"]:+.5f} $\\pm$ {last["coverage_difference_se"]:.5f}']
            ) + ' \\\\')
    with open(os.path.join(GENERATED, 'simpat_table_parity.tex'), 'w', encoding='UTF-8') as file:
        file.write('\n'.join(lines) + '\n')

    for config, tag in zip(configs, ('Four', 'Eight')):
        row = rows[(config, 0.1, 'float32', 1000)]  # benchmark setting: transmittance 0.1, single precision
        macro(f'ParityCoverageDiff{tag}', f'{row["coverage_difference"]:+.4f}')
        macro(f'ParityCoverageDiffSe{tag}', row['coverage_difference_se'], 4)
        macro(f'ParityMismatch{tag}', 100.0 * row['visibility_mismatch_rate'], 2)
        macro(f'ParityDiverged{tag}', 100.0 * rows[(config, 0.0, 'float32', 1000)]['diverged_target_share'], 1)
    macro('ParityEpisodes', str(rows[(configs[0], 0.0, 'float64', 1000)]['episodes']))

    def difference(a, b, paired=True):  # paired only when both samples come from the same episodes
        a, b = np.asarray(a), np.asarray(b)
        if paired and a.size == b.size:
            d = a - b
            return d.mean(), d.std(ddof=1) / np.sqrt(d.size), stats.ttest_rel(a, b).pvalue
        return a.mean() - b.mean(), np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size), stats.ttest_ind(a, b, equal_var=False).pvalue

    closed = {(r['config'], r['transmittance']): r for r in data.get('closed_loop', [])}
    for row in data.get('factorial', []):
        if row['transmittance'] != 0.1:
            continue
        tag = 'Four' if '4v8' in row['config'] else 'Eight'
        base = row['cameras-numpy_targets-numpy']
        for name, key in (('Camera', 'cameras-jax_targets-numpy'), ('Target', 'cameras-numpy_targets-jax')):
            mean, se, p = difference(row[key], base)
            macro(f'Port{name}Effect{tag}', f'{mean:+.3f}')
            macro(f'Port{name}EffectSe{tag}', se)
            macro(f'Port{name}EffectP{tag}', pvalue(p))
            print(f'port {row["config"]} {name}: {mean:+.4f} +- {se:.4f} p={p:.3g}')
        macro(f'PortOriginal{tag}', float(np.mean(base)))
        macro(f'PortBothOriginalEnv{tag}', float(np.mean(row['cameras-jax_targets-jax'])))
        jax_env = closed.get((row['config'], 0.1))
        if jax_env:
            # MATE-JAX draws its own episodes, independent of those of the original environment: Welch's test
            mean, se, p = difference(jax_env['jax_coverage'], row['cameras-jax_targets-jax'], paired=False)
            macro(f'PortBothJaxEnv{tag}', float(np.mean(jax_env['jax_coverage'])))
            print(f'simulator effect {row["config"]} (same ported agents, JAX env - original env): {mean:+.4f} +- {se:.4f} p={p:.3g}')


# 4. Decision latency on the idle device -------------------------------------------------------------- #

def latency_table():
    configs = ('4v8', '8v8', '8v16', '16v32')
    bench = {c: {r['variant']: np.asarray(r['ms_per_step_runs']) for r in load(f'paper/bench_idle_{c}.json') or []} for c in configs}
    if not all(bench.values()):
        print('latency: no results yet')
        return
    lines, spread = [], 0.0
    for variant, label in (('greedy', 'Rule-based cameras'), ('foresight', 'Option-level planning'),
                           ('traj-2', '\\FS, 2 synchronous rounds'), ('pub-4', '\\FS, published intents, shared memory'),
                           ('pub-4-hybrid', '\\FS, published intents, hybrid memory')):
        lines.append(' & '.join([label] + [f'{np.median(bench[c][variant]):.1f}' for c in configs]) + ' \\\\')
        spread = max([spread] + [bench[c][variant].max() / bench[c][variant].min() - 1.0 for c in configs])
    with open(os.path.join(GENERATED, 'simpat_table_latency.tex'), 'w', encoding='UTF-8') as file:
        file.write('\n'.join(lines) + '\n')
    for variant, name in (('foresight', 'Option'), ('traj-2', 'Sync'), ('pub-4', 'Pub'), ('pub-4-hybrid', 'PubHybrid')):
        for config, tag in (('4v8', 'Four'), ('16v32', 'Sixteen')):
            macro(f'IdleLatency{name}{tag}', float(np.median(bench[config][variant])), 1)
    macro('IdleLatencyRuns', str(len(bench['4v8']['traj-2'])))
    macro('IdleLatencySpread', float(np.ceil(100.0 * spread)), 0)
    print(f'latency: largest max/min - 1 over 7 runs = {100 * spread:.1f}%')


# 5. Exact optimum of two cameras ------------------------------------------------------------------ #

def pair_table():
    import json

    path = os.path.join(os.path.dirname(GENERATED), '..', 'mate', 'results', 'paper', 'exact_pair_4v8.json')
    rows = json.load(open(path, encoding='UTF-8'))
    names = {'greedy': 'Goal-directed', 'waypoint': 'RWP'}
    lines, below, above, share, relative, violations = [], [], [], [], [], 0
    for row in rows:
        lower, optimum, upper = (np.asarray(row[key]) for key in ('lower', 'optimum', 'upper'))
        violations += int(((optimum < lower - 1e-6) | (upper < optimum - 1e-6)).sum())
        loss, slack = optimum - lower, upper - optimum
        below.append(loss.mean())
        above.append(slack.mean())
        share.append(loss.mean() / (upper - lower).mean())
        relative.append(loss.mean() / optimum.mean())
        pair = '{} and {}'.format(*[index + 1 for index in row['pair']])
        lines.append(' & '.join([
            names[row['targets']], pair, f'{lower.mean():.4f}', f'{optimum.mean():.4f}', f'{upper.mean():.4f}',
            f'{loss.mean():.4f} $\\pm$ {loss.std(ddof=1) / np.sqrt(loss.size):.4f}', f'{loss.max():.4f}',
            f'{slack.mean():.4f}', f'{slack.max():.4f}',
        ]) + ' \\\\')
    with open(os.path.join(GENERATED, 'simpat_table_pair.tex'), 'w', encoding='UTF-8') as file:
        file.write('\n'.join(lines) + '\n')
    macro('PairEpisodes', str(len(rows[0]['lower'])))
    macro('PairLossMin', float(min(below)))
    macro('PairLossMax', float(max(below)))
    macro('PairSlackMax', float(max(above)), 4)
    macro('PairShareMin', float(np.floor(100.0 * min(share))), 0)
    macro('PairRelativeMin', 100.0 * float(min(relative)), 1)
    macro('PairRelativeMax', 100.0 * float(max(relative)), 1)
    macro('PairViolations', str(violations))
    print(f'exact pair: OPT-L {min(below):.4f}..{max(below):.4f}, U-OPT up to {max(above):.4f}, '
          f'share of the bracket below the optimum {100 * min(share):.1f}%+, violations {violations}')


# 6. Bounds and controllers under the same visibility rule (opaque obstacles) ----------------------- #

def opaque_macros():
    standard = rows_by(load('paper/main_4v8.json'))
    opaque = rows_by(load('paper/main_4v8_opaque.json'))
    bound = rows_by(load('paper/bound_dual_4v8_opaque.json'), key='mode')
    episodes = lambda row, key='per_episode_coverage': np.asarray(row[key])
    planner_loss, rule_loss, replay, above_upper = [], [], [], 0
    for target, tag in (('greedy', 'Goal'), ('waypoint', 'Rwp')):
        reference = bound[(target, 'clairvoyant-dp')]
        lower, upper = episodes(reference), episodes(reference, 'per_episode_dual_bound')
        replay.append(abs(reference['executed_coverage'] - reference['coverage']))
        for variant in ('foresight', 'traj-1', 'traj-2', 'traj-2-forecast', 'greedy'):
            loss = float((episodes(standard[(target, variant)]) - episodes(opaque[(target, variant)])).mean())
            (rule_loss if variant == 'greedy' else planner_loss).append(loss)
            above_upper += int((episodes(opaque[(target, variant)]) > upper + 1e-9).sum())
        option, coordinated, forecast = (episodes(opaque[(target, v)]).mean() for v in ('foresight', 'traj-2', 'traj-2-forecast'))
        macro(f'OpaqueLower{tag}', float(lower.mean()))
        macro(f'OpaqueUpper{tag}', float(upper.mean()))
        macro(f'OpaqueFc{tag}', float(forecast))
        macro(f'OpaqueGapLower{tag}', float(lower.mean() - forecast))
        macro(f'OpaqueGapUpper{tag}', float(upper.mean() - forecast))
        macro(f'OpaqueClosedFc{tag}', 100.0 * float((forecast - option) / (lower.mean() - option)), 0)
        macro(f'OpaqueClosedDual{tag}', 100.0 * float((forecast - option) / (upper.mean() - option)), 0)
        cell_upper = float(rows_by(load('paper/bound_cells_4v8.json'), key='mode')[(target, 'clairvoyant-dp')]['dual_bound'])
        macro(f'OpaqueClosedCell{tag}', 100.0 * float((forecast - option) / (cell_upper - option)), 0)
        macro(f'OpaqueFcLoss{tag}', float((episodes(standard[(target, 'traj-2-forecast')]) - episodes(opaque[(target, 'traj-2-forecast')])).mean()))
    macro('OpaqueReplayDiff', float(np.ceil(1e4 * max(replay)) / 1e4), 4)
    macro('OpaquePlannerLossMin', float(min(planner_loss)))
    macro('OpaquePlannerLossMax', float(max(planner_loss)))
    macro('OpaqueRuleLossMin', float(min(rule_loss)))
    macro('OpaqueRuleLossMax', float(max(rule_loss)))
    macro('OpaqueAboveUpper', str(above_upper))
    print(f'opaque: planners lose {min(planner_loss):.4f}..{max(planner_loss):.4f}, rule-based {min(rule_loss):.4f}..'
          f'{max(rule_loss):.4f}; replay difference {max(replay):.5f}; episodes above the Lagrangian bound {above_upper}')


# 7. Tables of the earlier version, with the unified names and without learned FORESIGHT ------------- #

RENAMES = [  # longest first
    ('MATE greedy cameras', 'Rule-based cameras'),
    ('FORESIGHT without learned forecaster', '\\FS, analytic forecasts'),
    ('Trajectories, 2 rounds (reference)', '\\FS, 2 rounds (reference)'),
]


def rename(line, extra=()):
    for old, new in list(extra) + RENAMES:
        if line.startswith(old):
            return new + line[len(old):]
    return line


def read_rows(name):
    with open(os.path.join(GENERATED, name), encoding='UTF-8') as file:
        return [line.rstrip('\n') for line in file if line.strip()]


def write_rows(name, lines):
    with open(os.path.join(GENERATED, name), 'w', encoding='UTF-8') as file:
        file.write('\n'.join(lines) + '\n')


def remark(lines):
    """Bold for the best and underline for the second best coverage of every column, among the controller rows above the
    first rule that do not use oracle information (rows that start with an indented '+')."""

    strip = lambda c: re.sub(r'\\(textbf|underline)\{([^}]*)\}', r'\2', c)
    end = next((i for i, line in enumerate(lines) if line.startswith('\\midrule')), len(lines))
    rows = [line[:-len(' \\\\')].split(' & ') for line in lines]
    controllers = [i for i in range(end) if not rows[i][0].startswith('\\quad') and not rows[i][0].startswith('\\multicolumn')]
    for c in range(1, len(rows[0])):
        values = []
        for i in controllers:
            rows[i][c] = strip(rows[i][c])
            try:
                values.append((float(rows[i][c]), i))
            except ValueError:
                pass
        ranked = sorted({v for v, _ in values}, reverse=True)
        for v, i in values:
            if v == ranked[0]:
                rows[i][c] = f'\\textbf{{{rows[i][c]}}}'
            elif len(ranked) > 1 and v == ranked[1]:
                rows[i][c] = f'\\underline{{{rows[i][c]}}}'
    return [(' & '.join(r) + ' \\\\') if not lines[k].startswith('\\midrule') else lines[k] for k, r in enumerate(rows)]


def derived_tables():
    bound = rows_by(load('paper/bound_dual_4v8.json'), key='mode')

    # Gap table: FORESIGHT variants named by their rounds and forecasts.
    gap = [rename(line, [('Trajectories, 1 round', '\\FS, 1 round, analytic forecasts'),
                         ('Trajectories, 2 rounds', '\\FS, 2 rounds, analytic forecasts'),
                         ('\\quad + learned forecaster', '\\FS, 2 rounds, learned forecaster'),
                         ('Hindsight bound (clairvoyant DP)', 'Hindsight bound (feasible clairvoyant schedule)'),
                         ('Lagrangian bound (clairvoyant, from above)', 'Lattice Lagrangian bound (upper bound on the lattice optimum)')])
           for line in read_rows('v2_table_gap.tex')]
    blurred = blur_oracle()
    paper = os.path.join(os.path.dirname(__file__), '..', 'mate', 'results', 'paper')
    blur8 = ['--', '--']
    if os.path.exists(os.path.join(paper, 'headroom_blur_8v8.json')):
        rows8 = {(r['targets'], r['mode']): r for r in json.load(open(os.path.join(paper, 'headroom_blur_8v8.json')))}
        blur8 = [f'{rows8[(t, "oracle-future-blur-0.5")]["coverage"]:.3f}' for t in TARGETS]
        for t in TARGETS:
            macro(f'BlurEight{SHORT[t]}', rows8[(t, 'oracle-future-blur-0.5')]['coverage'], 3)
            macro(f'BlurEightGain{SHORT[t]}', f'{rows8[(t, "oracle-future-blur-0.5")]["delta_vs_foresight"]:+.3f}')
    index = max(i for i, line in enumerate(gap) if line.startswith('\\quad + true future positions'))
    gap.insert(index + 1, ' & '.join(['\\quad + true future positions spread over half a cell']
                                     + [f'{blurred[t]:.3f}' for t in TARGETS] + blur8) + ' \\\\')
    if os.path.exists(os.path.join(paper, 'forecast_8v8.json')):
        rows8 = {(r['targets'], r['variant']): r for r in json.load(open(os.path.join(paper, 'forecast_8v8.json')))}
        i = next(i for i, line in enumerate(gap) if line.startswith('\\FS, 2 rounds, learned forecaster'))
        cells = gap[i][:-len(' \\\\')].split(' & ')
        cells[3:5] = [f'{rows8[(t, "traj-2-forecast")]["coverage"]:.3f}' for t in TARGETS]
        gap[i] = ' & '.join(cells) + ' \\\\'
        for t in TARGETS:
            r = rows8[(t, 'traj-2-forecast')]
            macro(f'FcEight{SHORT[t]}', r['coverage'], 3)
            macro(f'FcEightGain{SHORT[t]}', f'{r["paired_delta"]:+.3f}')
    i = next(i for i, line in enumerate(gap) if line.startswith('Lattice Lagrangian bound'))
    cell_upper = [rows_by(load(f'paper/bound_cells_{c}.json'), key='mode')[(t, 'clairvoyant-dp')]['dual_bound']
                  for c in ('4v8', '8v8') for t in TARGETS]
    gap.insert(i + 1, ' & '.join(['Lagrangian bound over cells (upper bound for continuous poses)']
                                 + [f'{v:.3f}' for v in cell_upper]) + ' \\\\')
    gap = remark(gap)
    write_rows('simpat_table_gap.tex', gap)

    # Main table: no Jain column (0.99--1.00 for every controller), no learned FORESIGHT, Lagrangian bound added.
    main = []
    for line in read_rows('v2_table_unified.tex'):
        if 'Learned' in line or 'DAgger' in line:
            continue
        line = line.replace('\\multicolumn{8}', '\\multicolumn{7}').replace('\\emph{Model-based}', '\\emph{Planners}')
        if not line.startswith('\\multicolumn') and not line.startswith('\\midrule'):
            cells = line[:-len(' \\\\')].split(' & ')
            assert len(cells) == 8, line
            jain = cells.pop(5)
            assert jain in ('--', '1.00', '0.99', '0.98') or jain.startswith('\\'), jain
            line = ' & '.join(cells) + ' \\\\'
        line = rename(line, [('Hindsight bound (clairvoyant)', 'Hindsight bound')])
        if line.startswith(('Hindsight bound', 'HiT-MAC')):
            line = line.replace(' & --', ' & n/a')
        main.append(line)
        if line.startswith('Hindsight bound'):
            main.append(' & '.join(['Lattice Lagrangian bound'] + [f'{bound[(t, "clairvoyant-dp")]["dual_bound"]:.3f}' for t in TARGETS]
                                   + ['n/a'] * 4) + ' \\\\')
            cell = rows_by(load('paper/bound_cells_4v8.json'), key='mode')
            main.append(' & '.join(['Lagrangian bound over cells'] + [f'{cell[(t, "clairvoyant-dp")]["dual_bound"]:.3f}' for t in TARGETS]
                                   + ['n/a'] * 4) + ' \\\\')
    write_rows('simpat_table_main.tex', main)

    # Ablation table: the hybrid memory equals the shared memory without loss (said in the text).
    ablation = [rename(line, [('Trajectories, 1 round', '\\FS, 1 round'), ('Trajectories, 3 rounds', '\\FS, 3 rounds'),
                              ('+ learned forecaster', '\\FS, 2 rounds, learned forecaster')])
                for line in read_rows('v2_table_ablation.tex') if not line.startswith('Hybrid memory')]
    ablation = [line.replace(' & --', ' & ref.') if '(reference)' in line else line for line in ablation]
    ablation.insert(1, zoom_control())
    write_rows('simpat_table_ablation.tex', ablation)

    # Scale table: no learned FORESIGHT; the share of the gap closed against both bounds.
    scale = [rename(line, [('Trajectories, 1 round', '\\FS, 1 round'), ('Trajectories, 2 rounds', '\\FS, 2 rounds'),
                           ('No communication', '\\FS, no communication'),
                           ('Published intents, every 4th step', '\\FS, published every 4th step'),
                           ('Lagrangian bound (from above)', 'Lattice Lagrangian bound'),
                           ('Gap closed (2 rounds)', 'Gap closed, hindsight bound')])
             for line in read_rows('v2_table_scale.tex') if not line.startswith('Learned')]
    cells, check = ['Gap closed, lattice Lagrangian bound'], []
    for config in ('4v8', '8v8', '8v16', '16v32'):
        rows = rows_by(load(f'paper/main_{config}.json'))
        if config in ('4v8', '8v8'):
            traj = rows_by(load(f'paper/headroom_traj2_{config}.json'), 'mode')
            for t in TARGETS:
                if (t, 'traj-2') not in rows and (t, 'foresight') in traj:
                    rows[(t, 'traj-2')] = traj[(t, 'foresight')]
        dual = rows_by(load(f'paper/bound_dual_{config}.json'), 'mode')
        for t in TARGETS:
            option, coordinated = rows[(t, 'foresight')]['coverage'], rows[(t, 'traj-2')]['coverage']
            lower, upper = dual[(t, 'clairvoyant-dp')]['coverage'], dual[(t, 'clairvoyant-dp')]['dual_bound']
            cells.append(f'{100.0 * (coordinated - option) / (upper - option):.0f}\\%')
            check.append(f'{100.0 * (coordinated - option) / (lower - option):.0f}\\%')
    closed_lower = next(line for line in scale if line.startswith('Gap closed, hindsight bound'))
    assert closed_lower[:-len(' \\\\')].split(' & ')[1:] == check, (closed_lower, check)
    scale.append(' & '.join(cells) + ' \\\\')
    write_rows('simpat_table_scale.tex', scale)
    print('derived tables written (gap, main, ablation, scale)')


def closed_table():
    """Share of the gap between option-level planning and each bound that FORESIGHT closes, both visibility rules."""

    settings = (('main_4v8.json', 'bound_dual_4v8.json'), ('main_4v8_opaque.json', 'bound_dual_4v8_opaque.json'))
    columns = []
    for controllers, bounds in settings:
        rows, bound = rows_by(load(f'paper/{controllers}')), rows_by(load(f'paper/{bounds}'), key='mode')
        cell = rows_by(load('paper/bound_cells_4v8.json'), key='mode')  # opaque obstacles, as the lattice bounds
        for t in TARGETS:
            option, full = rows[(t, 'foresight')]['coverage'], rows[(t, 'traj-2-forecast')]['coverage']
            lower, upper = bound[(t, 'clairvoyant-dp')]['coverage'], bound[(t, 'clairvoyant-dp')]['dual_bound']
            upper_cell = cell[(t, 'clairvoyant-dp')]['dual_bound']
            columns.append((option, full, lower, upper, upper_cell, 100.0 * (full - option) / (lower - option),
                            100.0 * (full - option) / (upper - option), 100.0 * (full - option) / (upper_cell - option)))
    labels = ['Option-level planning', '\\FS', 'Hindsight bound', 'Lattice Lagrangian bound', 'Lagrangian bound over cells']
    lines = [' & '.join([label] + [f'{c[i]:.3f}' for c in columns]) + ' \\\\' for i, label in enumerate(labels)]
    lines.append('\\midrule')
    for i, label in enumerate(['hindsight bound', 'lattice Lagrangian bound', 'bound over cells']):
        lines.append(' & '.join([f'Gap closed, {label}'] + [f'{c[5 + i]:.0f}\\%' for c in columns]) + ' \\\\')
    write_rows('simpat_table_closed.tex', lines)
    print('gap closed:', [f'{c[5]:.1f}/{c[6]:.1f}/{c[7]:.1f}' for c in columns])


def marl_figure():
    """Learning curves of the MARL baselines only (learned FORESIGHT moved to the appendix)."""

    import glob
    import json

    import make_figures_v2 as V

    initial = {}
    path = os.path.join(V.RESULTS, 'paper', 'initial_coverage.json')
    if os.path.exists(path):
        with open(path, encoding='UTF-8') as file:
            initial = json.load(file)
    runs = {}
    curves_ppo = 'curves_final' if glob.glob(os.path.join(V.RESULTS, 'curves_final', '*.jsonl')) else 'curves_long'
    for folder in (curves_ppo, 'curves_value_long'):
        for path in sorted(glob.glob(os.path.join(V.RESULTS, folder, '*.jsonl'))):
            name = os.path.basename(path)[:-6]
            algo, targets, _ = name.rsplit('_', 2)
            with open(path, encoding='UTF-8') as file:
                rows = [json.loads(line) for line in file if line.strip()]
            points = [(r['env_steps'], r['eval_coverage']) for r in rows if 'eval_coverage' in r]
            if name in initial:
                points = [(0, initial[name])] + points
            if points:
                runs.setdefault((algo, targets), []).append(points)
    main = rows_by(load('paper/main_4v8.json'))
    bound = rows_by(load('paper/bound_4v8.json'), 'mode')
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
    colors = plt.get_cmap('tab10')
    from matplotlib.lines import Line2D  # pylint: disable=import-outside-toplevel
    for ax, t in zip(axes, TARGETS):
        for index, (algo, label) in enumerate(V.MARL_LABELS):
            series = runs.get((algo, t))
            if not series:
                continue
            length = min(len(p) for p in series)
            x = np.asarray([p[0] for p in series[0][:length]]) / 1e6
            y = np.asarray([[v for _, v in p[:length]] for p in series])
            ax.plot(x, y.mean(0), color=colors(index), lw=1.3, label=label)
            ax.fill_between(x, y.mean(0) - y.std(0), y.mean(0) + y.std(0), color=colors(index), alpha=0.12, lw=0)
        for variant, ls, lw in (('traj-2-forecast', '-', 1.2), ('foresight', '--', 0.9), ('greedy', ':', 0.9)):
            ax.axhline(main[(t, variant)]['coverage'], color='black', ls=ls, lw=lw)
        ax.axhline(bound[(t, 'clairvoyant-dp')]['coverage'], color='#b22222', ls='-.', lw=0.9)
        ax.set_xlim(0, 100)
        ax.set_title(TARGET_TITLES[t])
        ax.set_xlabel('environment steps ($\\times 10^6$)')
    axes[0].set_ylabel('coverage rate')
    handles, _ = axes[0].get_legend_handles_labels()
    handles += [Line2D([], [], color='black', lw=1.2, label='FORESIGHT'),
                Line2D([], [], color='black', ls='--', lw=0.9, label='Option-level planning'),
                Line2D([], [], color='black', ls=':', lw=0.9, label='Rule-based cameras'),
                Line2D([], [], color='#b22222', ls='-.', lw=0.9, label='Hindsight bound')]
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(os.path.join(FIGURES, 'simpat_marl_curves.pdf'))
    plt.close(fig)


# 8. Oracle with blurred true future positions ------------------------------------------------------ #

def blur_oracle():
    rows = {(r['targets'], r['mode']): np.asarray(r['per_episode_coverage']) for r in load('paper/headroom_blur_4v8.json')}
    for t in TARGETS:
        tag = SHORT[t]
        base = rows[(t, 'foresight')]
        for mode, name in (('oracle-future', 'Point'), ('oracle-future-blur-0.5', 'BlurHalf'), ('oracle-future-blur-1', 'BlurOne')):
            d = rows[(t, mode)] - base
            macro(f'Oracle{name}{tag}', float(rows[(t, mode)].mean()))
            macro(f'Oracle{name}Gain{tag}', f'{d.mean():+.3f}')
            macro(f'Oracle{name}GainSe{tag}', float(d.std(ddof=1) / np.sqrt(d.size)))
        print(f'blur oracle {t}: point {rows[(t, "oracle-future")].mean() - base.mean():+.4f}, '
              f'half {rows[(t, "oracle-future-blur-0.5")].mean() - base.mean():+.4f}')
    return {t: rows[(t, 'oracle-future-blur-0.5')].mean() for t in TARGETS}


# 9. Zoom-grid control: option-level planning with the 31 zoom levels of FORESIGHT -------------------- #

def zoom_control():
    zoom = rows_by(load('paper/zoom_4v8.json'))
    main = rows_by(load('paper/main_4v8.json'))
    cells = ['Option-level planning, 31 zoom levels']
    for t in TARGETS:
        fine = np.asarray(zoom[(t, 'option-31')]['per_episode_coverage'])
        coarse = np.asarray(zoom[(t, 'foresight')]['per_episode_coverage'])
        assert np.allclose(coarse, main[(t, 'foresight')]['per_episode_coverage'], atol=1e-6), 'episodes are not paired'
        reference = np.asarray(main[(t, 'traj-2')]['per_episode_coverage'])
        d, g = fine - reference, fine - coarse
        cells += [f'{fine.mean():.3f}', f'{d.mean():+.3f} $\\pm$ {d.std(ddof=1) / np.sqrt(d.size):.3f}']
        tag = SHORT[t]
        macro(f'ZoomGain{tag}', f'{g.mean():+.3f}')
        macro(f'ZoomGainSe{tag}', float(g.std(ddof=1) / np.sqrt(g.size)))
        macro(f'ZoomCov{tag}', float(fine.mean()))
        one = np.asarray(main[(t, 'traj-1')]['per_episode_coverage'])
        macro(f'TrajOverZoom{tag}', f'{(one - fine).mean():+.3f}')
        print(f'zoom {t}: 31 levels {g.mean():+.4f} +- {g.std(ddof=1) / np.sqrt(g.size):.4f}; trajectories (1 round) over it {(one - fine).mean():+.4f}')
    return ' & '.join(cells) + ' \\\\'


def extended_tests():
    """The two smallest differences on 128 paired episodes: seed 0 (main_4v8) plus seeds 1-3 (main_4v8_seeds123)."""

    first, more = rows_by(load('paper/main_4v8.json')), rows_by(load('paper/main_4v8_seeds123.json'))
    tests = []
    for t in TARGETS:
        for name, a, b in (('Fc', 'traj-2-forecast', 'traj-2'), ('Third', 'traj-3', 'traj-2')):
            x = np.r_[first[(t, a)]['per_episode_coverage'], more[(t, a)]['per_episode_coverage']]
            y = np.r_[first[(t, b)]['per_episode_coverage'], more[(t, b)]['per_episode_coverage']]
            d = x - y
            tests.append({'t': t, 'name': name, 'd': d, 'p': stats.ttest_rel(x, y).pvalue,
                          'w': stats.wilcoxon(x, y).pvalue})
    for key in ('p', 'w'):
        for row, adjusted in zip(tests, holm([r[key] for r in tests])):
            row[key + '_holm'] = adjusted
    macro('ExtEpisodes', str(tests[0]['d'].size))
    macro('ExtNewEpisodes', str(tests[0]['d'].size - 32))
    macro('ExtPMax', pvalue(max(r['p_holm'] for r in tests)))
    macro('ExtWilcoxonPMax', pvalue(max(r['w_holm'] for r in tests)))
    for row in tests:
        tag, d = row['name'] + SHORT[row['t']], row['d']
        macro(f'Ext{tag}Gain', f'{d.mean():+.3f}')
        macro(f'Ext{tag}Se', float(d.std(ddof=1) / np.sqrt(d.size)))
        macro(f'Ext{tag}Wins', 100.0 * float((d > 0).mean()), 0)
        macro(f'Ext{tag}First', f'{d[:32].mean():+.3f}')
        macro(f'Ext{tag}New', f'{d[32:].mean():+.3f}')
        print(f'extended {row["t"]} {row["name"]}: {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(d.size):.4f} '
              f'(first 32 {d[:32].mean():+.4f}, new {d[32:].mean():+.4f}) p_holm={row["p_holm"]:.2g} w_holm={row["w_holm"]:.2g}')


def sensitivity_tables():
    """Planner parameters (generated/simpat_table_sens_model.tex) and scenario (generated/simpat_table_sens_env.tex)."""

    model = rows_by(load('paper/sens_model_4v8.json'))
    lines = [' & '.join(['Default ($H=50$, noise $\\times1$, $p_d=0.9$), CR']
                        + [f'{model[(t, "traj-2")]["coverage"]:.3f}' for t in TARGETS]) + ' \\\\', '\\midrule']
    labels = [('lookahead-10', 'Lookahead $H=10$'), ('lookahead-20', 'Lookahead $H=20$'),
              ('lookahead-30', 'Lookahead $H=30$'), ('lookahead-70', 'Lookahead $H=70$'),
              ('noise-half', 'Motion noise of the forecasts $\\times0.5$'),
              ('noise-double', 'Motion noise of the forecasts $\\times2$'),
              ('detect-0.7', 'Detection probability $p_d=0.7$'), ('detect-0.99', 'Detection probability $p_d=0.99$')]
    for variant, label in labels:
        cells = [label]
        for t in TARGETS:
            r = model[(t, variant)]
            cells.append(f'{r["paired_delta"]:+.3f} $\\pm$ {r["paired_delta_se"]:.3f}')
        lines.append(' & '.join(cells) + ' \\\\')
    write_rows('simpat_table_sens_model.tex', lines)
    for t in TARGETS:
        tag = SHORT[t]
        macro(f'SensNoiseHalf{tag}', f'{model[(t, "noise-half")]["paired_delta"]:+.3f}')
        macro(f'SensNoiseDouble{tag}', f'{model[(t, "noise-double")]["paired_delta"]:+.3f}')
        macro(f'SensLookTen{tag}', f'{model[(t, "lookahead-10")]["paired_delta"]:+.3f}')

    main = rows_by(load('paper/main_4v8.json'))
    default_bound = rows_by(load('paper/bound_dual_4v8.json'), key='mode')
    settings = [('Step 10~m', 'sens_speed10_4v8', 'bound_speed10_4v8'),
                ('Step 20~m (default)', None, None),
                ('Step 30~m', 'sens_speed30_4v8', 'bound_speed30_4v8'),
                ('Step 40~m', 'sens_speed40_4v8', 'bound_speed40_4v8'),
                ('Grid $16\\times16$', 'sens_grid16_4v8', None),
                ('Grid $24\\times24$', 'sens_grid24_4v8', None),
                ('Grid $48\\times48$', 'sens_grid48_4v8', None)]
    lines, over, gaps, grid_fs, grid_gain = [], {t: [] for t in TARGETS}, {t: [] for t in TARGETS}, {t: [] for t in TARGETS}, {t: [] for t in TARGETS}
    for label, name, bound_name in settings:
        rows = main if name is None else rows_by(load(f'paper/{name}.json'))
        if bound_name is not None:
            bound = rows_by(load(f'paper/{bound_name}.json'), key='mode')
        elif name is None:
            bound = default_bound
        else:
            bound = None
        if label.startswith('Grid $16'):
            lines.append('\\midrule')
        cells = [label]
        for t in TARGETS:
            option, fs = rows[(t, 'foresight')]['coverage'], rows[(t, 'traj-2')]['coverage']
            # The rule-based cameras and the bounds do not use belief maps: for other grids they are those of the default.
            rule = main[(t, 'greedy')] if label.startswith('Grid') else rows.get((t, 'greedy'))
            cells += [f'{rule["coverage"]:.3f}' if rule is not None else '--', f'{option:.3f}', f'{fs:.3f}']
            if bound is None and label.startswith('Grid'):
                b = default_bound[(t, 'clairvoyant-dp')]
                cells += [f'{b["coverage"]:.3f}', f'{b["dual_bound"]:.3f}']
            elif bound is not None:
                b = bound[(t, 'clairvoyant-dp')]
                cells += [f'{b["coverage"]:.3f}', f'{b["dual_bound"]:.3f}']
                if label.startswith('Step'):
                    over[t].append(fs - option)
                    gaps[t].append(b['coverage'] - fs)
            else:
                cells += ['--', '--']
            if label.startswith('Grid') or name is None:
                grid_fs[t].append(fs)
                grid_gain[t].append(fs - option)
        lines.append(' & '.join(cells) + ' \\\\')
    write_rows('simpat_table_sens_env.tex', lines)
    for t in TARGETS:
        tag = SHORT[t]
        macro(f'SensSpeedGainMin{tag}', min(over[t]))
        macro(f'SensSpeedGainMax{tag}', max(over[t]))
        macro(f'SensSpeedGapSlow{tag}', gaps[t][0])
        macro(f'SensSpeedGapFast{tag}', gaps[t][-1])
        macro(f'SensGridRange{tag}', max(grid_fs[t]) - min(grid_fs[t]))
        macro(f'SensGridGainMin{tag}', min(grid_gain[t]))
        macro(f'SensGridGainMax{tag}', max(grid_gain[t]))
        print(f'sensitivity {t}: FS over option by speed {np.round(over[t], 3)}, gap to hindsight {np.round(gaps[t], 3)}, '
              f'grid FS {np.round(grid_fs[t], 3)}, grid gain {np.round(grid_gain[t], 3)}')


# Reviewer checks: blurred oracle for option-level planning, random-action replay, three cameras, penalties --- #

def extra_checks():
    """Macros and the three-camera table for the checks added after review (Supplementary Sections S10 to S12)."""

    path = os.path.join(os.path.dirname(__file__), '..', 'mate', 'results', 'paper')

    # Blurred oracle for the option-level planner (headroom.py on the default planner).
    rows = {(r['targets'], r['mode']): r for r in json.load(open(os.path.join(path, 'headroom_blur_option_4v8.json')))}
    for t in TARGETS:
        tag = SHORT[t]
        for name, mode in (('Point', 'oracle-future'), ('BlurHalf', 'oracle-future-blur-0.5'), ('BlurOne', 'oracle-future-blur-1')):
            macro(f'Opt{name}Gain{tag}', f'{rows[(t, mode)]["delta_vs_foresight"]:+.3f}')
            macro(f'Opt{name}GainSe{tag}', rows[(t, mode)]['delta_se'])

    # Random-action replay (parity_multistep.py --actions random), 1000 steps.
    replay = [r for r in json.load(open(os.path.join(path, 'parity_random.json')))['trace_replay'] if r['steps'] == 1000]
    double = [r for r in replay if r['precision'] == 'float64']
    single = {('Four' if '4v8' in r['config'] else 'Eight'): r for r in replay if r['precision'] == 'float32'}
    macro('RandPosDouble', f'{max(r["max_position_error"] for r in double):.0e}'.replace('e-', '\\times10^{-') + '}')
    macro('RandCovDoubleMax', max(r['coverage_difference_max_abs'] for r in double), 4)
    macro('RandEpisodes', str(double[0]['episodes']))
    for tag, r in single.items():
        macro(f'RandPosSingle{tag}', r['max_position_error'], 0)
        macro(f'RandCovSingle{tag}', f'{r["coverage_difference"]:+.4f}')
        macro(f'RandCovSingleSe{tag}', r['coverage_difference_se'], 4)

    # Exact optimum for three cameras on the coarser lattice.
    lines, slack, slack_max, loss = [], [], [], []
    for r in json.load(open(os.path.join(path, 'exact_triple_4v8.json'))):
        lo, op, up = (np.asarray(r[k]) for k in ('lower', 'optimum', 'upper'))
        assert np.all(lo <= op + 1e-9) and np.all(op <= up + 1e-6), 'bracket violated'
        gain, gap = op - lo, up - op
        slack.append(gap.mean())
        slack_max.append(gap.max())
        loss.append(gain.mean())
        lines.append(' & '.join([
            'Goal-directed' if r['targets'] == 'greedy' else 'RWP', ', '.join(str(c + 1) for c in r['triple']),
            f'{lo.mean():.4f}', f'{op.mean():.4f}', f'{up.mean():.4f}',
            f'{gain.mean():.4f} $\\pm$ {gain.std(ddof=1) / np.sqrt(gain.size):.4f}', f'{gain.max():.4f}',
            f'{gap.mean():.4f}', f'{gap.max():.4f}']) + ' \\\\')
    with open(os.path.join(GENERATED, 'simpat_table_triple.tex'), 'w', encoding='UTF-8') as file:
        file.write('\n'.join(lines) + '\n')
    macro('TripleSlackMax', max(slack), 4)
    macro('TripleSlackEpisodeMax', max(slack_max), 4)
    macro('TripleLossMin', min(loss), 3)
    macro('TripleLossMax', max(loss), 3)
    print(f'exact triple: U-OPT mean up to {max(slack):.4f}, per episode up to {max(slack_max):.4f}; OPT-L {min(loss):.4f}..{max(loss):.4f}')

    # Penalized bounds (penalty_bound.py): tuning on seed-1 episodes, evaluation on seed-0 episodes (300 steps).
    names = {'lags': 'Future coverage of the pose', 'path': 'Value of the best path'}
    lines, best_gain = [], []
    for name in ('lags', 'path'):
        for r in json.load(open(os.path.join(path, f'penalty_{name}_4v8.json'))):
            base = r['tuning'][0]['upper']
            tuned = [x['upper'] for x in r['tuning'][1:]]
            best_gain.append(base - min(tuned))
            lines.append(' & '.join([
                names[name], 'Goal-directed' if r['targets'] == 'greedy' else 'RWP', str(len(tuned)),
                f'{base:.4f}', f'{min(tuned):.4f}', f'{max(tuned):.4f}', f'{r["lagrangian"]:.4f}', f'{r["penalized"]:.4f}',
            ]) + ' \\\\')
    with open(os.path.join(GENERATED, 'simpat_table_penalty.tex'), 'w', encoding='UTF-8') as file:
        file.write('\n'.join(lines) + '\n')
    macro('PenaltyBestGain', max(best_gain), 4)
    print(f'penalties: largest decrease on the tuning episodes {max(best_gain):.4f}')

    # Predictability of the targets given the full simulator state (predictability.py).
    for r in json.load(open(os.path.join(path, 'predictability_4v8.json'))):
        tag = SHORT[r['targets']]
        macro(f'PredSpreadTwenty{tag}', r['spread_mean_20'], 0)
        macro(f'PredSpreadTwentyMedian{tag}', r['spread_median_20'], 0)
        macro(f'PredSpreadFifty{tag}', r['spread_mean_50'], 0)
        macro(f'PredSplitTwenty{tag}', 100.0 * r['split_share_20'], 0)


# Continuous poses: Lagrangian bounds over cells, certified gaps, and where the width of the bracket comes from ----- #

CONFIG_TITLES = {'4v8': '4 cameras, 8 targets', '8v8': '8 cameras, 8 targets', '8v16': '8 cameras, 16 targets',
                 '16v32': '16 cameras, 32 targets'}


def cells_checks():
    path = os.path.join(os.path.dirname(__file__), '..', 'mate', 'results', 'paper')
    by_targets = lambda name: {r['targets']: r for r in json.load(open(os.path.join(path, name)))}
    lines, extra, widest, fine_gain = [], [], [], []
    for config in ('4v8', '8v8', '8v16', '16v32'):
        name = f'bound_cells_{config}.json'
        if not os.path.exists(os.path.join(path, name)):
            continue
        cells, lattice = by_targets(name), by_targets(f'bound_dual_{config}.json')
        fine_name = f'bound_cells2_{config}.json'
        cells2 = by_targets(fine_name) if os.path.exists(os.path.join(path, fine_name)) else None
        lines.append(f'\\multicolumn{{8}}{{l}}{{\\emph{{{CONFIG_TITLES[config]}}}}} \\\\')
        for t in TARGETS:
            low = np.asarray(cells[t]['per_episode_coverage'])
            up_lat = np.asarray(lattice[t]['per_episode_dual_bound'])
            up = np.asarray(cells[t]['per_episode_dual_bound'])
            width = up - low
            widest.append(width.mean())
            cells_fine = '--'
            if cells2 is not None:
                fine = np.asarray(cells2[t]['per_episode_dual_bound'])
                low2 = np.asarray(cells2[t]['per_episode_coverage'])
                cells_fine = f'{fine.mean():.3f}'
                if config == '4v8':
                    macro(f'CellFine{SHORT[t]}', fine.mean(), 3)
                    macro(f'CellFineWidth{SHORT[t]}', (fine - np.maximum(low, low2)).mean(), 3)
                    macro(f'LatFineLow{SHORT[t]}', low2.mean(), 3)
                fine_gain.append((up - fine).mean())
            lines.append(' & '.join(['\\quad ' + TARGET_TITLES[t].split(' ')[0] if t == 'greedy' else '\\quad RWP',
                                     f'{low.mean():.3f}', f'{up_lat.mean():.3f}', f'{up.mean():.3f}', cells_fine,
                                     ('0.000' if abs((up - up_lat).mean()) < 5e-4 else f'{(up - up_lat).mean():+.3f}'), f'{width.mean():.3f}', f'{width.max():.3f}']) + ' \\\\')
            if config == '4v8':
                tag = SHORT[t]
                macro(f'CellUpper{tag}', up.mean(), 3)
                macro(f'CellWidth{tag}', width.mean(), 3)
                macro(f'CellWidthMax{tag}', width.max(), 3)
                macro(f'CellExtra{tag}', f'{(up - up_lat).mean():+.3f}')
            extra.append((up - up_lat).mean())
    if lines:
        with open(os.path.join(GENERATED, 'simpat_table_cells.tex'), 'w', encoding='UTF-8') as file:
            file.write('\n'.join(lines) + '\n')
        macro('CellWidthMaxMean', max(widest), 3)
        macro('CellExtraMax', max(extra), 3)
        if fine_gain:
            macro('CellFineGainMax', max(fine_gain), 3)
        print(f'cells: mean widths {np.round(widest, 3)}, extra over the lattice bound {np.round(extra, 4)}')

    # Certified gaps of the controllers run in the simulator with opaque obstacles (four cameras).
    name = os.path.join(path, 'bound_cells_4v8.json')
    if os.path.exists(name):
        cells = by_targets('bound_cells_4v8.json')
        rows = {(r['targets'], r['variant']): r for r in json.load(open(os.path.join(path, 'main_4v8_opaque.json')))}
        violations, checked = 0, 0
        for (t, variant), r in rows.items():
            c = np.asarray(r['per_episode_coverage'])
            up = np.asarray(cells[t]['per_episode_dual_bound'])
            low = np.asarray(cells[t]['per_episode_coverage'])
            violations += int(np.sum(c > up + 1e-9))
            checked += c.size
            tag = {'traj-2-forecast': 'Fs', 'greedy': 'Rule', 'foresight': 'Opt'}.get(variant)
            if tag:
                gap = up - c
                macro(f'CertGap{tag}{SHORT[t]}', gap.mean(), 3)
                macro(f'CertGapSe{tag}{SHORT[t]}', gap.std(ddof=1) / np.sqrt(gap.size), 3)
                macro(f'CertLow{tag}{SHORT[t]}', np.maximum(low - c, 0.0).mean(), 3)
        macro('CertViolations', str(violations))
        macro('CertChecked', str(checked))
        print(f'certified: {violations} of {checked} controller episodes above the cell bound')

    # Width of the bracket for pairs: ascent, discretization, separation of the cameras.
    name = os.path.join(path, 'exact_pair_cells_4v8.json')
    if os.path.exists(name):
        lattice = {(r['targets'], tuple(r['pair'])): r for r in json.load(open(os.path.join(path, 'exact_pair_4v8.json')))}
        lines, parts = [], {'ascent': [], 'discretization': [], 'separation': []}
        for r in json.load(open(name)):
            key = (r['targets'], tuple(r['pair']))
            low, opt_cell, up = (np.asarray(r[k]) for k in ('lower', 'optimum', 'upper'))
            opt_lat = np.asarray(lattice[key]['optimum'])
            a, d, sep = opt_lat - low, opt_cell - opt_lat, up - opt_cell
            parts['ascent'].append(a.mean())
            parts['discretization'].append(d.mean())
            parts['separation'].append(sep.mean())
            lines.append(' & '.join(['Goal-directed' if r['targets'] == 'greedy' else 'RWP',
                                     ' and '.join(str(c + 1) for c in r['pair']), f'{low.mean():.4f}', f'{opt_lat.mean():.4f}',
                                     f'{opt_cell.mean():.4f}', f'{up.mean():.4f}', f'{a.mean():.4f}', f'{d.mean():.4f}',
                                     f'{sep.mean():.4f}']) + ' \\\\')
        with open(os.path.join(GENERATED, 'simpat_table_width.tex'), 'w', encoding='UTF-8') as file:
            file.write('\n'.join(lines) + '\n')
        for k, v in parts.items():
            macro(f'Width{k.capitalize()}Min', min(v), 4)
            macro(f'Width{k.capitalize()}Max', max(v), 4)
        print('pair width parts: ' + ', '.join(f'{k} {min(v):.4f}..{max(v):.4f}' for k, v in parts.items()))


def control_interval():
    """FORESIGHT with every command held for five steps, the control interval of the MARL baselines."""

    name = os.path.join(os.path.dirname(__file__), '..', 'mate', 'results', 'paper', 'control_interval_4v8.json')
    if not os.path.exists(name):
        return
    rows = {(r['targets'], r['variant']): r for r in json.load(open(name))}
    for t in TARGETS:
        r = rows[(t, 'traj-2-forecast-repeat-5')]
        tag = SHORT[t]
        macro(f'RepeatFs{tag}', r['coverage'], 3)
        macro(f'RepeatDelta{tag}', f'{r["paired_delta"]:+.3f}')
        macro(f'RepeatDeltaSe{tag}', r['paired_delta_se'], 3)
        print(f'control interval {t}: {r["coverage"]:.4f} ({r["paired_delta"]:+.4f} +- {r["paired_delta_se"]:.4f})')
    # Row of the main table (network metrics for goal-directed targets, as the other rows).
    g = rows[('greedy', 'traj-2-forecast-repeat-5')]['per_camera']
    n = len(g['camera_bytes'])
    row = ' & '.join(['\\quad every command held for 5 steps']
                     + [f'{rows[(t, "traj-2-forecast-repeat-5")]["coverage"]:.3f}' for t in TARGETS]
                     + [f'{sum(g["camera_unique_coverage"]):.2f}', f'{rows[("greedy", "traj-2-forecast-repeat-5")]["redundancy"]:.2f}',
                        f'{(sum(g["camera_rotation"]) + sum(g["camera_zoom"])) / n:.2f}', f'{sum(g["camera_bytes"]) / n:.0f}']) + ' \\\\'
    main = read_rows('simpat_table_main.tex')
    i = next(i for i, line in enumerate(main) if line.startswith('FORESIGHT &'))
    main.insert(i + 1, row)
    write_rows('simpat_table_main.tex', main)


def main():
    paired_table()
    burst_figure()
    parity_table()
    latency_table()
    pair_table()
    opaque_macros()
    derived_tables()
    closed_table()
    marl_figure()
    extended_tests()
    sensitivity_tables()
    extra_checks()
    cells_checks()
    control_interval()
    with open(os.path.join(GENERATED, 'simpat_macros.tex'), 'w', encoding='UTF-8') as file:
        file.write('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in sorted(MACROS.items())) + '\n')
    for key, value in sorted(MACROS.items()):
        print(f'  \\{key} = {value}')


if __name__ == '__main__':
    main()
