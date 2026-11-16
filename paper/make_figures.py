"""Generate the tables (generated/*.tex) and figures (figures/*.pdf) of the JNCA manuscript.

Inputs are the JSON files written by ``mate/research/eval_foresight.py``. Network metrics that are
not measured inside the simulator (packet sizes, radio energy) are computed from the protocol and
radio models stated in Section 3 of the paper, using the measured detection counts.

    python3 paper-jnca/make_figures.py
"""

import json
import os
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get('MATE_RESULTS', os.path.join(HERE, '..', 'mate', 'results'))
GENERATED = os.path.join(HERE, 'generated')
FIGURES = os.path.join(HERE, 'figures')

NUM_TARGETS = 8  # MATE-4v8-9
NUM_CAMERAS = 4
GRID_CELLS = 32 * 32

# Protocol model (Section 3.4): bytes per packet = header 4 + pose 4 + intent 2 + 5 per detection.
PACKET_FIXED_BYTES = 10
PACKET_BYTES_PER_DETECTION = 5
MAP_SHARING_BYTES = NUM_TARGETS * GRID_CELLS * 2  # all per-target belief maps in float16
OBSERVATION_SHARING_BYTES = 126 * 4  # raw MATE camera observation in float32

# First-order radio model (Heinzelman et al., 2000).
E_ELEC = 50e-9  # J / bit
EPS_AMP = 100e-12  # J / bit / m^2
BROADCAST_DISTANCE = np.hypot(1300.0, 1300.0)  # farthest teammate (cameras in opposite quadrants)

TARGET_LABELS = {'greedy': 'Goal-directed targets', 'waypoint': 'Random Waypoint targets'}
TARGET_SHORT = {'greedy': 'goal-directed', 'waypoint': 'RWP'}


def num(x, digits=1):
    """Thousands separated by a thin space (LaTeX)."""

    return f'{x:,.{digits}f}'.replace(',', '\\,')


def load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return []
    with open(path, encoding='UTF-8') as file:
        return json.load(file)


def index(rows):
    return {(r['targets'], r['variant']): r for r in rows}


def paired(a, b):
    d = np.asarray(a['per_episode_coverage']) - np.asarray(b['per_episode_coverage'])
    return d.mean(), d.std(ddof=1) / np.sqrt(d.size)


def detections_per_camera(row):
    return float(np.mean(row['per_camera']['camera_coverage'])) * NUM_TARGETS


def packet_bytes(row, communicates=True):
    if not communicates:
        return 0.0
    return PACKET_FIXED_BYTES + PACKET_BYTES_PER_DETECTION * detections_per_camera(row)


def radio_energy_mj(bytes_sent, bytes_received_total):
    bits_tx, bits_rx = 8.0 * bytes_sent, 8.0 * bytes_received_total
    energy = bits_tx * (E_ELEC + EPS_AMP * BROADCAST_DISTANCE**2) + bits_rx * E_ELEC
    return 1e3 * energy


def jain(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.sum() ** 2 / (values.size * np.square(values).sum()))


def fmt(x, digits=3):
    return f'{x:.{digits}f}'


def write(name, text):
    os.makedirs(GENERATED, exist_ok=True)
    with open(os.path.join(GENERATED, name), 'w', encoding='UTF-8') as file:
        file.write(text)


def drop_of(variant):
    match = re.search(r'drop([0-9.]+)$', variant)
    return float(match.group(1)) if match else 0.0


# --------------------------------------------------------------------------------------------- #


def main_table(main):
    variants = [
        ('greedy', 'Greedy cameras (MATE)', True),
        ('foresight', 'FORESIGHT', True),
        ('refine', 'FORESIGHT + refinement', True),
        ('no-communication', 'FORESIGHT, no communication', False),
    ]
    lines = []
    for targets in ('greedy', 'waypoint'):
        if (targets, 'greedy') not in main:
            continue
        reference = main[(targets, 'greedy')]
        lines.append(rf'\multicolumn{{9}}{{l}}{{\emph{{{TARGET_LABELS[targets]}}}}} \\')
        for key, label, communicates in variants:
            row = main.get((targets, key))
            if row is None:
                continue
            delta, se = paired(row, reference)
            per_camera = row['per_camera']
            unique = float(np.sum(per_camera['camera_unique_coverage']))
            rotation = float(np.mean(per_camera['camera_rotation']))
            zoom = float(np.mean(per_camera['camera_zoom']))
            bytes_text = '--' if key == 'greedy' else f'{packet_bytes(row, communicates):.1f}'
            delta_text = '--' if key == 'greedy' else f'{delta:+.3f} $\\pm$ {se:.3f}'
            lines.append(
                f'{label} & {fmt(row["coverage"])} & {delta_text} & {fmt(row["real_coverage"])} & '
                f'{fmt(unique)} & {fmt(row["redundancy"])} & {fmt(jain(per_camera["camera_coverage"]))} & '
                f'{rotation:.2f} / {zoom:.2f} & {bytes_text} \\\\'
            )
        lines.append(r'\midrule')
    if lines and lines[-1] == r'\midrule':
        lines.pop()
    write('table_main.tex', '\n'.join(lines) + '\n')


def loss_rows(main, sweep):
    """{targets: {architecture: [(p, coverage, se_vs_foresight)]}}"""

    rows = {**main, **sweep}
    out = {}
    for targets in ('greedy', 'waypoint'):
        reference = rows.get((targets, 'foresight'))
        if reference is None:
            continue
        curves = {'team': [(0.0, reference['coverage'])], 'hybrid': []}
        for (t, variant), row in rows.items():
            if t != targets:
                continue
            for arch in ('team', 'hybrid'):
                if variant.startswith(arch + '-drop'):
                    curves[arch].append((drop_of(variant), row['coverage']))
        for arch in curves:
            curves[arch] = sorted(set(curves[arch]))
        out[targets] = curves
    return out


def loss_table(main, sweep):
    curves = loss_rows(main, sweep)
    drops = sorted({p for c in curves.values() for arch in c.values() for p, _ in arch})
    header = ' & '.join(f'{int(round(p * 100))}\\%' for p in drops)
    lines = [f'Packet loss rate $p$ & {header} \\\\', r'\midrule']
    for targets, c in curves.items():
        for arch, label in (('team', 'Shared memory'), ('hybrid', 'Hybrid memory')):
            values = dict(c[arch])
            cells = ' & '.join(fmt(values[p]) if p in values else '--' for p in drops)
            lines.append(f'{label} ({TARGET_SHORT[targets]}) & {cells} \\\\')
        none = main.get((targets, 'no-communication'))
        if none is not None:
            lines.append(
                f'No communication ({TARGET_SHORT[targets]}) & '
                f'\\multicolumn{{{len(drops)}}}{{c}}{{{fmt(none["coverage"])} (independent of $p$)}} \\\\'
            )
        lines.append(r'\midrule')
    lines.pop()
    write('table_loss.tex', '\n'.join(lines) + '\n')
    write(
        'table_loss_wrapper.tex',
        f'\\begin{{tabular}}{{l{"c" * len(drops)}}}\n\\toprule\n\\inputbody{{generated/table_loss.tex}}\n\\bottomrule\n\\end{{tabular}}\n',
    )


def paired_table(main):
    pairs = [
        ('refine', 'foresight', 'Refinement $-$ FORESIGHT'),
        ('hybrid-drop0.5', 'team-drop0.5', 'Hybrid $-$ shared memory ($p=0.5$)'),
        ('hybrid-drop0.9', 'team-drop0.9', 'Hybrid $-$ shared memory ($p=0.9$)'),
        ('foresight', 'no-communication', 'Shared memory $-$ no communication ($p=0$)'),
    ]
    lines = []
    for a, b, label in pairs:
        cells = []
        for targets in ('greedy', 'waypoint'):
            if (targets, a) in main and (targets, b) in main:
                d = np.asarray(main[(targets, a)]['per_episode_coverage']) - np.asarray(
                    main[(targets, b)]['per_episode_coverage']
                )
                se = d.std(ddof=1) / np.sqrt(d.size)
                cells.append(f'{d.mean():+.3f} $\\pm$ {se:.3f} & {100 * np.mean(d > 0):.0f}\\%')
            else:
                cells.append('-- & --')
        lines.append(f'{label} & {" & ".join(cells)} \\\\')
    write('table_paired.tex', '\n'.join(lines) + '\n')


def overhead_table(main):
    row = main.get(('greedy', 'foresight'))
    if row is None:
        return
    ours = packet_bytes(row)
    entries = [
        ('FORESIGHT packet (pose + intent + detections)', ours),
        ('Raw observation sharing', OBSERVATION_SHARING_BYTES),
        ('Belief-map sharing (8 maps, $32\\times32$, float16)', MAP_SHARING_BYTES),
    ]
    lines = []
    for label, nbytes in entries:
        received = (NUM_CAMERAS - 1) * nbytes
        lines.append(
            f'{label} & {num(nbytes)} & {num(nbytes / ours, 0)}$\\times$ & '
            f'{num(radio_energy_mj(nbytes, received))} \\\\'
        )
    write('table_overhead.tex', '\n'.join(lines) + '\n')
    write(
        'macros.tex',
        '\n'.join(
            [
                f'\\newcommand{{\\PacketBytes}}{{{ours:.1f}}}',
                f'\\newcommand{{\\MapBytes}}{{{num(MAP_SHARING_BYTES, 0)}}}',
                f'\\newcommand{{\\MapRatio}}{{{MAP_SHARING_BYTES / ours:.0f}}}',
                f'\\newcommand{{\\ObsRatio}}{{{OBSERVATION_SHARING_BYTES / ours:.0f}}}',
                f'\\newcommand{{\\BroadcastDistance}}{{{BROADCAST_DISTANCE:.0f}}}',
                f'\\newcommand{{\\PacketEnergy}}{{{radio_energy_mj(ours, (NUM_CAMERAS - 1) * ours):.1f}}}',
            ]
        )
        + '\n',
    )


# Figures ------------------------------------------------------------------------------------- #


def style():
    plt.rcParams.update(
        {
            'font.family': 'serif',
            'font.size': 9,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.25,
            'legend.frameon': False,
            'figure.dpi': 200,
        }
    )


def loss_figure(main, sweep):
    curves = loss_rows(main, sweep)
    if not curves:
        return
    fig, axes = plt.subplots(1, len(curves), figsize=(7.0, 2.6), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, (targets, c) in zip(axes, curves.items()):
        for arch, label, marker, color in (
            ('team', 'Shared memory', 'o', '#c0392b'),
            ('hybrid', 'Hybrid memory', 's', '#1f6fb2'),
        ):
            if c[arch]:
                p, cov = zip(*c[arch])
                ax.plot(np.asarray(p) * 100, cov, marker=marker, label=label, color=color, lw=1.6, ms=4)
        for key, label, style_, color in (
            ('greedy', 'Greedy cameras', '--', '#555555'),
            ('no-communication', 'No communication', ':', '#555555'),
        ):
            row = main.get((targets, key))
            if row is not None:
                ax.axhline(row['coverage'], ls=style_, color=color, lw=1.0, label=label)
        ax.set_title(TARGET_LABELS[targets])
        ax.set_xlabel('Packet loss rate (%)')
        ax.set_ylabel('Coverage rate')
    axes[0].legend(loc='lower left', fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'packet_loss.pdf'))
    plt.close(fig)


def per_camera_figure(main):
    variants = [
        ('greedy', 'Greedy'),
        ('foresight', 'FORESIGHT'),
        ('no-communication', 'No comm.'),
        ('team-drop0.9', 'Shared, $p$=0.9'),
        ('hybrid-drop0.9', 'Hybrid, $p$=0.9'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, targets in zip(axes, ('greedy', 'waypoint')):
        x = 0.0
        ticks, labels = [], []
        for key, label in variants:
            row = main.get((targets, key))
            if row is None:
                continue
            per_camera = row['per_camera']
            unique = np.asarray(per_camera['camera_unique_coverage'])
            shared = np.asarray(per_camera['camera_coverage']) - unique
            positions = x + np.arange(unique.size) * 0.18
            ax.bar(positions, unique, width=0.16, color='#1f6fb2', label='Unique' if x == 0 else None)
            ax.bar(positions, shared, width=0.16, bottom=unique, color='#a9c8e8', label='Redundant' if x == 0 else None)
            ticks.append(positions.mean())
            labels.append(label)
            x += unique.size * 0.18 + 0.35
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=7.5)
        ax.set_title(TARGET_LABELS[targets])
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('Per-camera coverage')
    axes[0].legend(loc='upper left', fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'per_camera.pdf'))
    plt.close(fig)


def objective_figure():
    # Paired study with 32 episodes (Section 6.2); the numbers come from research/README.md.
    rows = [
        ('window', 0.040, 0.008),
        ('no forecast\n(window)', 0.050, 0.008),
        ('no reach time\n(window)', 0.058, 0.010),
        ('point forecast\n(window)', 0.053, 0.008),
        ('no negative\ninformation (window)', 0.025, 0.009),
        ('path, $\\gamma$=0.98', 0.082, 0.008),
        ('path, $\\gamma$=0.95', 0.091, 0.007),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    labels, values, errors = zip(*rows)
    colors = ['#a9c8e8'] * 5 + ['#1f6fb2'] * 2
    ax.bar(range(len(rows)), values, yerr=errors, color=colors, capsize=2)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([label.replace('\n', ' ') for label in labels], fontsize=7, rotation=18, ha='right')
    ax.set_ylabel('$\\Delta$ coverage vs. greedy')
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'objective_ablation.pdf'))
    plt.close(fig)


def ablation_table(ablation):
    variants = [
        ('foresight', 'FORESIGHT (path objective, full)'),
        ('no-forecast', 'w/o forecast (current belief only)'),
        ('no-reach', 'w/o reach time (poses reached instantly)'),
        ('no-negative-info', 'w/o negative information'),
        ('point-forecast', 'Point forecast ($\\sigma = 0$)'),
        ('window', 'Window objective instead of path'),
    ]
    lines = []
    for key, label in variants:
        cells = []
        for targets in ('greedy', 'waypoint'):
            row, ref = ablation.get((targets, key)), ablation.get((targets, 'greedy'))
            if row is None or ref is None:
                cells.append('\\emph{pending} & --')
                continue
            delta, se = paired(row, ref)
            cells.append(f'{fmt(row["coverage"])} & {delta:+.3f} $\\pm$ {se:.3f}')
        lines.append(f'{label} & {" & ".join(cells)} \\\\')
    write('table_ablation.tex', '\n'.join(lines) + '\n')


FINAL_WINDOW_ENV_STEPS = 2 * 512 * 500 * 5  # two training episodes of all 512 environments


def load_curves():
    """{(algo, targets): {seed: list of JSON metrics}} from results/curves/*.jsonl."""

    import glob  # pylint: disable=import-outside-toplevel

    curves = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, 'curves', '*.jsonl'))):
        match = re.match(r'(\w+)_(\w+)_s(\d+)\.jsonl', os.path.basename(path))
        if not match:
            continue
        algo, targets, seed = match.group(1), match.group(2), int(match.group(3))
        with open(path, encoding='UTF-8') as file:
            rows = [json.loads(line) for line in file if line.strip()]
        if rows:
            curves.setdefault((algo, targets), {})[seed] = rows
    return curves


def _stack(runs, key):
    """Common x (environment steps) and (seeds, points) y for the rows that have ``key``."""

    series = []
    for rows in runs.values():
        points = [(r['env_steps'], r[key]) for r in rows if key in r and np.isfinite(r[key])]
        if points:
            series.append(points)
    if not series:
        return None, None
    length = min(len(points) for points in series)
    x = np.asarray([p[0] for p in series[0][:length]], dtype=np.float64)
    y = np.asarray([[p[1] for p in points[:length]] for points in series], dtype=np.float64)
    return x, y


def _episode_mean(x, y, window=FINAL_WINDOW_ENV_STEPS // 2):
    """Trailing mean over one training episode of all environments (removes the episode-phase ripple)."""

    out = np.empty_like(y)
    for i in range(y.shape[-1]):
        mask = (x > x[i] - window) & (x <= x[i])
        out[..., i] = y[..., mask].mean(-1)
    return out


def _ema(y, alpha=0.3):
    out = np.empty_like(y)
    out[..., 0] = y[..., 0]
    for i in range(1, y.shape[-1]):
        out[..., i] = alpha * y[..., i] + (1.0 - alpha) * out[..., i - 1]
    return out


ALGO_STYLE = {
    'mappo': ('MAPPO', '#c0392b'),
    'ippo': ('IPPO', '#8e44ad'),
    'mappofs': ('MAPPO + FORESIGHT', '#d35400'),
    'ippofs': ('IPPO + FORESIGHT', '#2c3e72'),
}


def learning_curves(main):
    curves = load_curves()
    if not curves:
        write('table_learned.tex', '\\multicolumn{4}{c}{\\todo{learning curves pending}} \\\\\n')
        return
    for key, filename, ylabel, smooth in (
        ('eval_coverage', 'learning_curves.pdf', 'Coverage rate (evaluation)', False),
        ('episode_coverage', 'training_curves.pdf', 'Coverage rate (training episodes)', True),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=False)
        for ax, targets in zip(axes, ('greedy', 'waypoint')):
            for algo, (label, color) in ALGO_STYLE.items():
                runs = curves.get((algo, targets))
                if not runs:
                    continue
                x, y = _stack(runs, key)
                if x is None:
                    continue
                if smooth:
                    y = _ema(y)
                mean, std = y.mean(0), y.std(0)
                ax.plot(x / 1e6, mean, color=color, lw=1.6, label=f'{label} ({y.shape[0]} seeds)')
                ax.fill_between(x / 1e6, mean - std, mean + std, color=color, alpha=0.18, lw=0)
            for variant, label, style_, color in (
                ('refine', 'FORESIGHT + refinement', '-', '#0b3d91'),
                ('foresight', 'FORESIGHT', '--', '#1f6fb2'),
                ('greedy', 'Greedy cameras', ':', '#555555'),
            ):
                row = main.get((targets, variant))
                if row is not None:
                    ax.axhline(row['coverage'], ls=style_, color=color, lw=1.3, label=label)
            ax.set_title(TARGET_LABELS[targets])
            ax.set_xlabel('Environment steps ($\\times 10^6$)')
            ax.set_ylabel(ylabel)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=7, frameon=False)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        fig.savefig(os.path.join(FIGURES, filename))
        plt.close(fig)

    # Final-performance table: last evaluation of every run.
    lines = []
    for targets in ('greedy', 'waypoint'):
        for algo, (label, _) in ALGO_STYLE.items():
            runs = curves.get((algo, targets))
            if not runs:
                continue
            finals = [
                [r['eval_coverage'] for r in rows if 'eval_coverage' in r][-1]
                for rows in runs.values()
                if any('eval_coverage' in r for r in rows)
            ]
            steps = max(rows[-1]['env_steps'] for rows in runs.values())
            if finals:
                lines.append(
                    f'{label} ({TARGET_SHORT[targets]}) & {np.mean(finals):.3f} $\\pm$ {np.std(finals):.3f} & '
                    f'{len(finals)} & {steps / 1e6:.0f}M \\\\'
                )
    write('table_learned.tex', '\n'.join(lines) + '\n')


# (backbone, label, colour, FORESIGHT partner run names in priority order)
BACKBONES = [('mappo', 'MAPPO', '#d35400', ('hrl', 'mappofs')), ('ippo', 'IPPO', '#2c3e72', ('hrlippo', 'ippofs'))]


def paired_learning_curves():
    """Backbone vs. backbone + FORESIGHT learning curves, in the style of the group's drafts."""

    curves = load_curves()
    partners = {name for _, _, _, names in BACKBONES for name in names}
    if not any(algo in partners for algo, _ in curves):
        return
    from matplotlib.ticker import FuncFormatter  # pylint: disable=import-outside-toplevel

    for key, filename, ylabel in (
        ('coverage_rate', 'paired_curves.pdf', 'coverage rate (training)'),
        ('eval_coverage', 'paired_curves_eval.pdf', 'evaluation coverage rate'),
    ):
        targets_list = [t for t in ('greedy', 'waypoint') if any((b, t) in curves for b, _, _, _ in BACKBONES)]
        fig, axes = plt.subplots(
            len(targets_list), len(BACKBONES), figsize=(3.4 * len(BACKBONES), 2.7 * len(targets_list)),
            squeeze=False, sharey='row',
        )
        for row, targets in enumerate(targets_list):
            for col, (backbone, label, color, partner_names) in enumerate(BACKBONES):
                ax = axes[row, col]
                partner = next((name for name in partner_names if (name, targets) in curves), None)
                for algo, style_, name in ((backbone, '--', label), (partner, '-', f'{label} + FORESIGHT')):
                    runs = curves.get((algo, targets)) if algo else None
                    if not runs:
                        continue
                    x, y = _stack(runs, key)
                    if x is None:
                        continue
                    y = _episode_mean(x, y) if key == 'coverage_rate' else y
                    mean, std = 100 * y.mean(0), 100 * y.std(0)
                    ax.plot(x / 1e6, mean, style_, color=color, lw=2.0 if algo != backbone else 1.8, label=name)
                    ax.fill_between(x / 1e6, mean - std, mean + std, color=color, alpha=0.18, lw=0)
                ax.set_title(f'{label} ({TARGET_SHORT[targets]} targets)', color=color, fontsize=10)
                ax.set_xlabel('environment steps')
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}M' if v else '0'))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}%'))
                if col == 0:
                    ax.set_ylabel(ylabel)
                ax.legend(loc='center', fontsize=7.5)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES, filename))
        plt.close(fig)


HRL_COMM_COST, HRL_ACTUATION_COST = 0.1, 0.05  # lambda_c, lambda_a of the training runs


def hrl_results():
    """Final comparison table (learned high levels vs. rule-based transmission) and tx-rate curves."""

    curves = load_curves()
    # Rules evaluated over the training horizon (500 decisions) when available, else over 1000 environment steps.
    rules = load('hrl_rules_4v8_h500.json') or load('hrl_rules_4v8.json')
    if not rules and not any(algo.startswith('hrl') for algo, _ in curves):
        write('table_hrl.tex', '\\multicolumn{7}{c}{\\todo{FORESIGHT-HRL results pending}} \\\\\n')
        return
    packet_energy = radio_energy_mj(PACKET_FIXED_BYTES + PACKET_BYTES_PER_DETECTION * 2.0, 0.0)  # mJ per packet

    def reward(cov, tx, act):
        """Per-step reward of eq. (reward): coverage minus transmission and actuation costs."""

        return f'{cov - HRL_COMM_COST * tx - HRL_ACTUATION_COST * act:.3f}'

    def final(runs, key):
        """Mean over the last two training episodes of every seed, then mean / std over seeds.

        All training environments are reset together, so with targets that never end an episode (RWP)
        the per-update coverage follows the phase of the episode; whole episodes average that out.
        """

        values = []
        for rows in runs.values():
            points = [(r['env_steps'], r[key]) for r in rows if key in r and np.isfinite(r[key])]
            if points:
                last = points[-1][0]
                values.append(np.mean([v for step, v in points if step > last - FINAL_WINDOW_ENV_STEPS]))
        return (np.mean(values), np.std(values)) if values else (np.nan, np.nan)

    lines = []
    names = {
        'mappo': 'Flat MAPPO',
        'ippo': 'Flat IPPO',
        'hrl': 'FORESIGHT-HRL (MAPPO)',
        'hrlippo': 'FORESIGHT-HRL (IPPO)',
        'hrlnotx': 'FORESIGHT-HRL, options only',
    }
    rule_names = {
        'always': 'Greedy, always transmit',
        'event': 'Greedy, event-triggered',
        'periodic-4': 'Greedy, periodic (1/4)',
        'never': 'Greedy, never transmit',
    }
    for targets in ('greedy', 'waypoint'):
        block = []
        for rule, label in rule_names.items():
            row = next((r for r in rules if r['targets'] == targets and r['rule'] == rule), None)
            if row is not None:
                block.append(
                    f'{label} & {row["coverage_rate"]:.3f} & {row["tx_rate"]:.2f} & {row["collision_rate"]:.2f} & '
                    f'{row["actuation"]:.2f} & {row["tx_rate"] * 5 * packet_energy:.0f} & '
                    f'{reward(row["coverage_rate"], row["tx_rate"], row["actuation"])} \\\\'
                )
        for algo, label in names.items():
            runs = curves.get((algo, targets))
            if not runs:
                continue
            cov, cov_sd = final(runs, 'coverage_rate')
            tx, _ = final(runs, 'tx_rate')
            act, _ = final(runs, 'actuation')
            tx = 0.0 if not np.isfinite(tx) else tx
            act_text = f'{act:.2f}' if np.isfinite(act) else '--'
            # The flat baselines are trained on coverage alone, so their cost-adjusted reward is not comparable.
            reward_text = reward(cov, tx, act) if algo.startswith('hrl') else '--'
            block.append(
                f'{label} & {cov:.3f} $\\pm$ {cov_sd:.3f} & {tx:.2f} & -- & {act_text} & '
                f'{tx * 5 * packet_energy:.0f} & {reward_text} \\\\'
            )
        if block:
            lines.append(f'\\multicolumn{{7}}{{l}}{{\\emph{{{TARGET_LABELS[targets]}}}}} \\\\')
            lines.extend(block)
            lines.append('\\midrule')
    if lines and lines[-1] == '\\midrule':
        lines.pop()
    write('table_hrl.tex', '\n'.join(lines) + '\n')

    # Transmission rate of the learned high level over training.
    runs_by_target = {t: curves.get(('hrl', t)) for t in ('greedy', 'waypoint')}
    if not any(runs_by_target.values()):
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, (targets, runs) in zip(axes, runs_by_target.items()):
        if runs:
            for key, label, color in (('tx_rate', 'transmission rate', '#1f6fb2'), ('coverage_rate', 'coverage rate', '#d35400')):
                x, y = _stack(runs, key)
                if x is None:
                    continue
                y = _episode_mean(x, y)
                ax.plot(x / 1e6, y.mean(0), color=color, lw=1.8, label=f'FORESIGHT-HRL {label}')
                ax.fill_between(x / 1e6, y.mean(0) - y.std(0), y.mean(0) + y.std(0), color=color, alpha=0.18, lw=0)
        for rule, style_ in (('always', ':'), ('periodic-4', '-.'), ('event', '--')):
            row = next((r for r in rules if r['targets'] == targets and r['rule'] == rule), None)
            if row is not None:
                ax.axhline(row['tx_rate'], ls=style_, color='#555555', lw=1.0, label=f'{rule} (tx rate)')
        ax.set_title(TARGET_LABELS[targets])
        ax.set_xlabel('Environment steps ($\\times 10^6$)')
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel('rate')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=6.5, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(os.path.join(FIGURES, 'hrl_comm.pdf'))
    plt.close(fig)


def main():
    os.makedirs(FIGURES, exist_ok=True)
    style()
    main_rows = index(load('foresight_new_proposals_4v8.json'))
    sweep_rows = index(load('foresight_loss_sweep_4v8.json'))
    main_table(main_rows)
    ablation_table(index(load('foresight_ablation_path_4v8.json')))
    loss_table(main_rows, sweep_rows)
    paired_table(main_rows)
    overhead_table(main_rows)
    loss_figure(main_rows, sweep_rows)
    per_camera_figure(main_rows)
    objective_figure()
    learning_curves(main_rows)
    paired_learning_curves()
    hrl_results()
    print('wrote', sorted(os.listdir(GENERATED)), sorted(os.listdir(FIGURES)))


if __name__ == '__main__':
    main()
