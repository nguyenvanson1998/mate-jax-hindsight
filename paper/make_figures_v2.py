"""Tables (generated/v2_*.tex), macros and figures of the hindsight-guided manuscript.

Inputs (mate/results): paper/*.json (eval_foresight.py, headroom.py, clairvoyant.py, bench_foresight.py on 32
paired episodes), comm/eval*.json (eval_comm.py), comm/curves/*.jsonl (communication MARL training),
policy_eval/*.json (learned controllers), forecaster_*.json, hindsight/*.jsonl, terminal_probe_greedy.json.
Missing inputs produce TODO placeholders, so the manuscript always compiles.

    python3 paper-jnca/make_figures_v2.py
"""

import glob
import json
import os
import statistics

import numpy as np

from make_figures import (  # noqa: F401  (shared models and helpers of the first manuscript)
    BROADCAST_DISTANCE, FIGURES, GENERATED, MAP_SHARING_BYTES, OBSERVATION_SHARING_BYTES, RESULTS, jain, load,
    plt, radio_energy_mj, style,
)


TARGETS = ('greedy', 'waypoint')
SHORT = {'greedy': 'Goal', 'waypoint': 'Rwp'}

# Message model of Trajectories: detection packet = header (id, sequence, count) 4 B + pose 4 B + 5 B per
# detection; trajectory intent = header 2 B + start pose 2 B + one 4-bit lattice move per horizon (25 moves).
DETECTION_FIXED_BYTES = 8
DETECTION_BYTES_PER_TARGET = 5
INTENT_BYTES = 2 + 2 + 13
OPTION_PACKET_FIXED_BYTES = 10  # Option-level: the packet also carries a 2-byte option intent


def todo(text):
    return f'\\todo{{{text}}}'


def write(name, text):
    os.makedirs(GENERATED, exist_ok=True)
    with open(os.path.join(GENERATED, name), 'w', encoding='UTF-8') as file:
        file.write(text)


def rows_by(rows, key='variant'):
    return {(r['targets'], r.get(key) or r.get('mode')): r for r in rows}


def paired(a, b):
    d = np.asarray(a['per_episode_coverage']) - np.asarray(b['per_episode_coverage'])
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(d.size)), float((d > 0).mean())


def cov(row):
    return f'{row["coverage"]:.3f}' if row else '--'


def energy_mj(bytes_per_step, receivers=3):
    return radio_energy_mj(bytes_per_step, receivers * bytes_per_step)


class Macros:
    def __init__(self):
        self.values = {}

    def __setitem__(self, name, value):
        self.values[name] = value

    def get(self, name, value, digits=3):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            self.values[name] = todo(name)
        elif isinstance(value, str):
            self.values[name] = value
        else:
            self.values[name] = f'{value:.{digits}f}'

    def write(self):
        lines = [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in sorted(self.values.items())]
        write('v2_macros.tex', '\n'.join(lines) + '\n')


M = Macros()


# Gap decomposition ------------------------------------------------------------------------------ #


def gap_table():
    main4 = rows_by(load('paper/main_4v8.json'))
    main8 = rows_by(load('paper/main_8v8.json'))
    option = {c: rows_by(load(f'paper/headroom_option_{c}.json'), 'mode') for c in ('4v8', '8v8')}
    traj = {c: rows_by(load(f'paper/headroom_traj2_{c}.json'), 'mode') for c in ('4v8', '8v8')}
    bound = {c: rows_by(load(f'paper/bound_{c}.json'), 'mode') for c in ('4v8', '8v8')}
    columns = [(c, t) for c in ('4v8', '8v8') for t in TARGETS]

    def value(source, key):
        row = source.get(key)
        return row['coverage'] if row else None

    lines = []

    def add(label, getter, bold=False):
        cells = []
        for config, targets in columns:
            v = getter(config, targets)
            cells.append('--' if v is None else (f'\\textbf{{{v:.3f}}}' if bold else f'{v:.3f}'))
        lines.append(f'{label} & ' + ' & '.join(cells) + ' \\\\')

    add('MATE greedy cameras', lambda c, t: value(main4 if c == '4v8' else main8, (t, 'greedy')))
    add('Option-level planning', lambda c, t: value(option[c], (t, 'foresight')))
    add('\\quad + true current positions', lambda c, t: value(option[c], (t, 'oracle-now')))
    add('\\quad + true future positions', lambda c, t: value(option[c], (t, 'oracle-future')))
    add('Trajectories, 1 round', lambda c, t: value(main4 if c == '4v8' else main8, (t, 'traj-1')))
    add('Trajectories, 2 rounds', lambda c, t: value(traj[c], (t, 'foresight')), bold=True)
    add('\\quad + true future positions', lambda c, t: value(traj[c], (t, 'oracle-future')))
    add('\\quad + learned forecaster', lambda c, t: value(main4, (t, 'traj-2-forecast')) if c == '4v8' else None)
    lines.append('\\midrule')
    add('Hindsight bound (clairvoyant DP)', lambda c, t: value(bound[c], (t, 'clairvoyant-dp')))
    dual = {c: rows_by(load(f'paper/bound_dual_{c}.json'), 'mode') for c in ('4v8', '8v8')}
    add('Lagrangian bound (clairvoyant, from above)', lambda c, t: dual[c].get((t, 'clairvoyant-dp'), {}).get('dual_bound'))
    add('Teleport bound (no rate limits)', lambda c, t: option[c].get((t, 'foresight'), {}).get('teleport'))
    add('Visibility ceiling', lambda c, t: option[c].get((t, 'foresight'), {}).get('ceiling'))
    write('v2_table_gap.tex', '\n'.join(lines) + '\n')

    for config, targets in columns:
        tag = SHORT[targets] + ('Eight' if config == '8v8' else '')
        o, t2 = value(option[config], (targets, 'foresight')), value(traj[config], (targets, 'foresight'))
        b = value(bound[config], (targets, 'clairvoyant-dp'))
        M.get(f'O{tag}', o)
        M.get(f'TTwo{tag}', t2)
        M.get(f'Bound{tag}', b)
        d = dual[config].get((targets, 'clairvoyant-dp'), {}).get('dual_bound')
        M.get(f'Dual{tag}', d)
        M.get(f'DualGap{tag}', (d - b) if None not in (d, b) else None)
        if config == '4v8':
            fc = value(main4, (targets, 'traj-2-forecast'))
            M.get(f'ClosedDual{tag}', 100.0 * (fc - o) / (d - o) if None not in (fc, o, d) else None, digits=0)
            g = value(main4, (targets, 'greedy'))
            M.get(f'GapGreedy{tag}', (b - g) if None not in (b, g) else None, digits=2)
            by_algo = {}
            for r in load('paper/learned_4v8_final.json') or load('paper/learned_4v8_long.json') or load('paper/learned_4v8.json'):
                if r['targets'] == targets:
                    by_algo.setdefault(r['variant'], []).append(r['coverage'])
            best = max(float(np.mean(v)) for v in by_algo.values()) if by_algo else None
            M.get(f'GapMarl{tag}', (b - best) if None not in (b, best) else None, digits=2)
        M.get(f'ONow{tag}', value(option[config], (targets, 'oracle-now')))
        M.get(f'OFut{tag}', value(option[config], (targets, 'oracle-future')))
        M.get(f'TFut{tag}', value(traj[config], (targets, 'oracle-future')))
        row = option[config].get((targets, 'foresight'), {})
        M.get(f'Tele{tag}', row.get('teleport'))
        M.get(f'Ceil{tag}', row.get('ceiling'))
        exec_row = bound[config].get((targets, 'clairvoyant-dp'))
        M.get(f'BoundExec{tag}', exec_row.get('executed_coverage') if exec_row else None)
        closed = 100.0 * (t2 - o) / (b - o) if None not in (o, t2, b) else None
        M.get(f'ClosedT{tag}', closed, digits=0)
        if config == '4v8':
            fc = value(main4, (targets, 'traj-2-forecast'))
            closed_fc = 100.0 * (fc - o) / (b - o) if None not in (o, fc, b) else None
            M.get(f'ClosedFc{tag}', closed_fc, digits=0)


# Main comparison -------------------------------------------------------------------------------- #

MAIN_VARIANTS = [
    ('greedy', 'MATE greedy cameras', 'option'),
    ('foresight', 'Option-level, shared memory', 'option'),
    ('hybrid', 'Option-level, hybrid memory', 'option'),
    ('traj-1', 'Trajectories, 1 round', 'traj1'),
    ('traj-2', 'Trajectories, 2 rounds', 'traj2'),
    ('traj-3', 'Trajectories, 3 rounds', 'traj3'),
    ('traj-2-hybrid', 'Trajectories, 2 rounds, hybrid', 'traj2'),
    ('traj-2-none', 'Trajectories, no communication', 'none'),
    ('traj-2-forecast', 'Trajectories + learned forecaster', 'traj2'),
]


def message_bytes(row, kind):
    """Bytes per camera and step of the coordination messages (detections + intents)."""

    if kind == 'none':
        return 0.0
    detections = float(np.mean(row['per_camera']['camera_coverage'])) * 8
    if kind == 'option':
        return OPTION_PACKET_FIXED_BYTES + DETECTION_BYTES_PER_TARGET * detections
    rounds = {'traj1': 1, 'traj2': 2, 'traj3': 3, 'pub': 1}[kind]
    return DETECTION_FIXED_BYTES + DETECTION_BYTES_PER_TARGET * detections + rounds * INTENT_BYTES


def main_table():
    main = rows_by(load('paper/main_4v8.json'))
    lines = []
    for variant, label, kind in MAIN_VARIANTS:
        cells = [label]
        for targets in TARGETS:
            row, ref = main.get((targets, variant)), main.get((targets, 'foresight'))
            if row is None:
                cells += ['--', '--']
                continue
            d, se, _ = paired(row, ref)
            cells += [f'{row["coverage"]:.3f}', '--' if variant == 'foresight' else f'{d:+.3f}']
        row = main.get(('greedy', variant))
        if row is None:
            cells += ['--'] * 5
        else:
            pc = row['per_camera']
            effort = float(np.mean(pc['camera_rotation']) + np.mean(pc['camera_zoom']))
            cells += [
                f'{float(np.sum(pc["camera_unique_coverage"])):.2f}', f'{row["redundancy"]:.2f}',
                f'{jain(pc["camera_coverage"]):.2f}', f'{effort:.2f}', f'{message_bytes(row, kind):.0f}',
            ]
        lines.append(' & '.join(cells) + ' \\\\')
    write('v2_table_main.tex', '\n'.join(lines) + '\n')

    for targets in TARGETS:
        tag = SHORT[targets]
        ref = main.get((targets, 'foresight'))
        for variant, name in (('greedy', 'Greedy'), ('traj-1', 'TOne'), ('traj-3', 'TThree'), ('traj-2-hybrid', 'THyb'),
                              ('traj-2-none', 'TNone'), ('traj-2-forecast', 'TFc'), ('traj-2', 'TTwoMain')):
            row = main.get((targets, variant))
            M.get(f'{name}{tag}', row['coverage'] if row else None)
        t2 = main.get((targets, 'traj-2'))
        if t2 and ref:
            d, se, wins = paired(t2, ref)
            M.get(f'GainT{tag}', d)
            M.get(f'GainTSe{tag}', se)
            M.get(f'GainTWins{tag}', 100 * wins, digits=0)
        fc = main.get((targets, 'traj-2-forecast'))
        if fc and t2:
            d, se, wins = paired(fc, t2)
            M.get(f'GainFc{tag}', d)
            M.get(f'GainFcSe{tag}', se)
            M.get(f'GainFcWins{tag}', 100 * wins, digits=0)
        none = main.get((targets, 'traj-2-none'))
        if none and t2:
            M.get(f'CommValue{tag}', t2['coverage'] - none['coverage'])
        g = main.get((targets, 'greedy'))
        if g and t2:
            M.get(f'GainTGreedy{tag}', t2['coverage'] - g['coverage'])
    t2 = main.get(('greedy', 'traj-2'))
    if t2:
        M.get('BytesTTwo', message_bytes(t2, 'traj2'), digits=0)
        M.get('BytesPub', message_bytes(t2, 'pub'), digits=0)
        M.get('EnergyPub', energy_mj(message_bytes(t2, 'pub')), digits=0)
    o = main.get(('greedy', 'foresight'))
    if o:
        M.get('BytesO', message_bytes(o, 'option'), digits=0)
    M.get('IntentBytes', float(INTENT_BYTES), digits=0)


def scale_table():
    main8 = rows_by(load('paper/main_8v8.json'))
    bound = rows_by(load('paper/bound_8v8.json'), 'mode')
    traj = rows_by(load('paper/headroom_traj2_8v8.json'), 'mode')
    labels = [('greedy', 'MATE greedy cameras'), ('foresight', 'Option-level, shared memory'),
              ('hybrid', 'Option-level, hybrid memory'), ('traj-1', 'Trajectories, 1 round'),
              ('traj-2', 'Trajectories, 2 rounds'), ('traj-2-hybrid', 'Trajectories, 2 rounds, hybrid')]
    lines = []
    for variant, label in labels:
        cells = [label]
        for targets in TARGETS:
            row = main8.get((targets, variant))
            if row is None and variant == 'traj-2':
                row = traj.get((targets, 'foresight'))
            cells.append(f'{row["coverage"]:.3f}' if row else todo('8v8 running'))
        lines.append(' & '.join(cells) + ' \\\\')
    lines.append('\\midrule')
    lines.append('Hindsight bound & ' + ' & '.join(cov(bound.get((t, 'clairvoyant-dp'))) for t in TARGETS) + ' \\\\')
    write('v2_table_scale.tex', '\n'.join(lines) + '\n')


# Packet loss ------------------------------------------------------------------------------------ #


def loss_results():
    sweep = rows_by(load('paper/loss_4v8.json'))
    main = rows_by(load('paper/main_4v8.json'))
    series = {
        'Option-level, shared': ('foresight-team-{p}', ('foresight',)),
        'Option-level, hybrid': ('foresight-hybrid-{p}', ('hybrid',)),
        'Trajectories, shared': ('traj-team-{p}', ('traj-2',)),
        'Trajectories, hybrid': ('traj-hybrid-{p}', ('traj-2-hybrid', 'traj-2')),
    }
    losses = (0.0, 0.25, 0.5, 0.75, 0.9)
    data = {}
    for targets in TARGETS:
        for name, (pattern, zero) in series.items():
            values = []
            for p in losses:
                row = None
                if p == 0.0:
                    row = next((main.get((targets, v)) for v in zero if main.get((targets, v))), None)
                else:
                    row = sweep.get((targets, pattern.format(p=p)))
                values.append(row['coverage'] if row else np.nan)
            data[(targets, name)] = values
    lines = []
    for name in series:
        cells = [name]
        for targets in TARGETS:
            cells += ['--' if not np.isfinite(v) else f'{v:.3f}' for v in data[(targets, name)][1:]]
        lines.append(' & '.join(cells) + ' \\\\')
    write('v2_table_loss.tex', '\n'.join(lines) + '\n')
    for targets in TARGETS:
        for name, macro in (('Option-level, shared', 'LossOShared'), ('Option-level, hybrid', 'LossOHybrid'),
                            ('Trajectories, shared', 'LossTShared'), ('Trajectories, hybrid', 'LossTHybrid')):
            M.get(f'{macro}{SHORT[targets]}', data[(targets, name)][-1])

    if not sweep:
        return
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=False)
    colors = {'Option-level, shared': '#999999', 'Option-level, hybrid': '#555555',
              'Trajectories, shared': '#e08b2f', 'Trajectories, hybrid': '#1f6fb2'}
    for ax, targets in zip(axes, TARGETS):
        for name in series:
            ax.plot(losses, data[(targets, name)], marker='o', ms=3.5, lw=1.6, color=colors[name],
                    ls='--' if 'shared' in name else '-', label=name)
        none = main.get((targets, 'traj-2-none'))
        if none:
            ax.axhline(none['coverage'], color='#1f6fb2', ls=':', lw=1.0, label='Trajectories, no communication')
        ax.set_title({'greedy': 'Goal-directed targets', 'waypoint': 'Random Waypoint targets'}[targets])
        ax.set_xlabel('packet loss rate $p$')
    axes[0].set_ylabel('coverage rate')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=7)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(os.path.join(FIGURES, 'v2_packet_loss.pdf'))
    plt.close(fig)


# Decentralized publication of intents ---------------------------------------------------------- #


def published_table():
    rows = rows_by(load('paper/published_4v8.json'))
    labels = [('traj-2', 'Synchronous rounds (2 per step)', 2.0), ('pub-always', 'Published every step', 1.0),
              ('pub-event-1', 'Published on change (event)', None), ('pub-period-4', 'Published every 4th step', 0.25),
              ('pub-always-hybrid', 'Published every step, hybrid memory', 1.0)]
    lines = []
    for variant, label, rate in labels:
        cells = [label]
        for targets in TARGETS:
            row, ref = rows.get((targets, variant)), rows.get((targets, 'traj-2'))
            if row is None:
                cells += [todo('running'), '--']
                continue
            d, se, _ = paired(row, ref)
            cells += [f'{row["coverage"]:.3f}', '--' if variant == 'traj-2' else f'{d:+.3f} $\\pm$ {se:.3f}']
        cells.append('--' if rate is None else f'{rate * INTENT_BYTES:.0f}')
        lines.append(' & '.join(cells) + ' \\\\')
    write('v2_table_published.tex', '\n'.join(lines) + '\n')
    for targets in TARGETS:
        for variant, name in (('pub-always', 'PubAlways'), ('pub-period-4', 'PubPeriod'), ('pub-event-1', 'PubEvent')):
            row = rows.get((targets, variant))
            M.get(f'{name}{SHORT[targets]}', row['coverage'] if row else None)


# Communication control (rules vs MARL) --------------------------------------------------------- #

RULE_LABELS = {'always': 'Always', 'never': 'Never', 'event': 'Event-triggered', 'periodic-4': 'Periodic (TDMA, 1/4)',
               'event-slot': 'Event-triggered in own slot'}
LEARNER_LABELS = {'mappo-h': 'MAPPO, hindsight critic', 'mappo': 'MAPPO', 'ippo': 'IPPO'}
DETECTION_PACKET_BYTES = 18  # mean detection packet (8 B + 5 B x 2 detections)


def comm_tables():
    rows = load('comm/eval.json') or (load('comm/eval_rules.json') + load('comm/eval_phase_greedy.json')
                                       + load('comm/eval_phase_greedy_nohindsight.json'))
    if not rows:
        write('v2_table_comm.tex', '\\multicolumn{7}{c}{' + todo('communication results pending') + '} \\\\\n')
        return
    e_packet, e_intent = energy_mj(DETECTION_PACKET_BYTES), energy_mj(INTENT_BYTES)
    lines = []
    for targets in TARGETS:
        block = []
        for rule, label in RULE_LABELS.items():
            r = next((x for x in rows if x['targets'] == targets and x['method'] == rule and 'mode' not in x), None)
            if r:
                energy = r['tx_rate'] * e_packet + r['intent_rate'] * e_intent
                block.append(f'{label} & {r["reward"]:.3f} & {r["coverage_rate"]:.3f} & {r["tx_rate"]:.2f} & '
                             f'{r["intent_rate"]:.2f} & {r["collision_rate"]:.2f} & {energy:.0f} \\\\')
        for learner, label in LEARNER_LABELS.items():
            runs = [x for x in rows if x['targets'] == targets and x.get('mode') == 'stochastic'
                    and x['method'].startswith(learner + '_')]
            if not runs:
                continue
            def stat(key):
                values = [x[key] for x in runs]
                return statistics.mean(values), (statistics.pstdev(values) if len(values) > 1 else 0.0)
            reward, reward_sd = stat('reward')
            energy = statistics.mean(x['tx_rate'] * e_packet + x['intent_rate'] * e_intent for x in runs)
            block.append(f'{label} & {reward:.3f} $\\pm$ {reward_sd:.3f} & {stat("coverage_rate")[0]:.3f} & '
                         f'{stat("tx_rate")[0]:.2f} & {stat("intent_rate")[0]:.2f} & {stat("collision_rate")[0]:.2f} & '
                         f'{energy:.0f} \\\\')
            M.get(f'Comm{learner.replace("-h", "H").replace("mappo", "Mappo").replace("ippo", "Ippo")}{SHORT[targets]}', reward)
        if block:
            lines.append(f'\\multicolumn{{7}}{{l}}{{\\emph{{{"Goal-directed" if targets == "greedy" else "Random Waypoint"} targets}}}} \\\\')
            lines += block
            lines.append('\\midrule')
        rules = [x for x in rows if x['targets'] == targets and 'mode' not in x]
        if rules:
            best = max(rules, key=lambda x: x['reward'])
            M.get(f'CommBestRule{SHORT[targets]}', best['reward'])
            M[f'CommBestRuleName{SHORT[targets]}'] = RULE_LABELS.get(best['method'], best['method']).lower()
    if lines and lines[-1] == '\\midrule':
        lines.pop()
    write('v2_table_comm.tex', '\n'.join(lines) + '\n')
    M.get('EnergyPacket', e_packet, digits=0)
    M.get('EnergyIntent', e_intent, digits=0)


def comm_curves():
    files = sorted(glob.glob(os.path.join(RESULTS, 'comm', 'curves', '*.jsonl')))
    if not files:
        return
    runs = {}
    for path in files:
        name = os.path.basename(path)[:-6]
        learner, targets, _ = name.rsplit('_', 2)
        with open(path, encoding='UTF-8') as file:
            rows = [json.loads(line) for line in file if line.strip()]
        if rows:
            runs.setdefault((learner, targets), []).append(rows)
    style()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True)
    colors = {'mappo-h': '#1f6fb2', 'mappo': '#e08b2f', 'ippo': '#6a3d9a'}
    episode = 3 * 128 * 100  # decisions of all environments in three training episodes (they reset together)
    for col, targets in enumerate(TARGETS):
        for learner, label in LEARNER_LABELS.items():
            series = runs.get((learner, targets))
            if not series:
                continue
            length = min(len(r) for r in series)
            decisions = np.asarray([r['agent_steps'] for r in series[0][:length]], dtype=np.float64)
            x = decisions * 5 / 1e6  # environment steps (frame skip 5)
            for row, key, scale in ((0, 'reward', 1 / 5), (1, 'tx_rate', 1.0)):
                y = np.asarray([[r[key] * scale for r in rows[:length]] for rows in series])
                smooth = np.stack([y[:, (decisions > d - episode) & (decisions <= d)].mean(-1) for d in decisions], -1)
                ax = axes[row, col]
                ax.plot(x, smooth.mean(0), color=colors[learner], lw=1.6, label=label)
                ax.fill_between(x, smooth.mean(0) - smooth.std(0), smooth.mean(0) + smooth.std(0),
                                color=colors[learner], alpha=0.18, lw=0)
        axes[0, col].set_title({'greedy': 'Goal-directed targets', 'waypoint': 'Random Waypoint targets'}[targets])
        axes[1, col].axhline(0.25, color='#555555', ls='--', lw=1.0, label='periodic schedule (1/4)')
        axes[1, col].set_xlabel('environment steps ($\\times 10^6$)')
    axes[0, 0].set_ylabel('reward per step')
    axes[1, 0].set_ylabel('packets per camera-step')
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=7)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(FIGURES, 'v2_comm_curves.pdf'))
    plt.close(fig)


# Learning in hindsight: forecaster, distillation, terminal values ----------------------------- #


def learning_table():
    lines = []
    for targets in TARGETS:
        f1, f2 = load(f'forecaster_{targets}.json'), load(f'forecaster_{targets}_r2.json')
        if isinstance(f1, dict) and f1.get('history'):
            best = min(f1['history'], key=lambda h: h['held_out_nll'])
            M.get(f'NllAnalytic{SHORT[targets]}', best['analytic_nll'], digits=2)
            M.get(f'NllLearned{SHORT[targets]}', best['held_out_nll'], digits=2)
            M.get(f'FcOneBase{SHORT[targets]}', f1['trajectory']['coverage'])
            M.get(f'FcOneLearned{SHORT[targets]}', f1['trajectory+learned-forecast']['coverage'])
            M.get(f'FcOneGain{SHORT[targets]}', f1['paired_delta']['mean'])
            M.get(f'FcOneGainSe{SHORT[targets]}', f1['paired_delta']['se'])
        if isinstance(f2, dict) and f2.get('paired_delta'):
            M.get(f'FcTwoGain{SHORT[targets]}', f2['paired_delta']['mean'])
            M.get(f'FcTwoGainSe{SHORT[targets]}', f2['paired_delta']['se'])

    # Policy distillation from the hindsight teacher (goal-directed targets, 16 episodes).
    def last_eval(path):
        if not os.path.exists(path):
            return None
        with open(path, encoding='UTF-8') as file:
            rows = [json.loads(line) for line in file if line.strip()]
        evals = [r for r in rows if 'eval_coverage' in r]
        reference = next((r['reference'] for r in rows if 'reference' in r), None)
        return (evals[-1]['eval_coverage'] if evals else None), reference

    value, reference = last_eval(os.path.join(RESULTS, 'hindsight', 'greedy_value.jsonl')) or (None, None)
    waypoint, _ = last_eval(os.path.join(RESULTS, 'hindsight', 'greedy_wp.jsonl')) or (None, None)
    move, _ = last_eval(os.path.join(RESULTS, 'hindsight', 'greedy_proto.jsonl')) or (None, None)
    M.get('DistillValue', value)
    M.get('DistillWaypoint', waypoint)
    M.get('DistillMove', move)
    if reference:
        M.get('DistillTeacher', reference.get('teacher_closed_loop'))
        M.get('DistillPrior', reference.get('initial_student'))
    probe = load('terminal_probe_greedy.json')
    if probe:
        base = next(r for r in probe if r['weight'] == 0.0)
        best = max((r for r in probe if r['weight'] > 0), key=lambda r: r['delta'])
        M.get('TerminalBase', base['coverage'])
        M.get('TerminalBestDelta', best['delta'])
        M.get('TerminalBestSe', best['delta_se'])


# Learned controllers on raw control (MARL baselines) ------------------------------------------ #


def marl_table():
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS, 'policy_eval', '*.json'))):
        with open(path, encoding='UTF-8') as file:
            rows += json.load(file)
    planners = rows_by(load('paper/learned_protocol_4v8.json'))
    labels = [('mappo', 'MAPPO, raw pan/zoom'), ('ippo', 'IPPO, raw pan/zoom'), ('hrl', 'FORESIGHT-HRL (MAPPO)'),
              ('hrlippo', 'FORESIGHT-HRL (IPPO)')]
    lines = []
    for algo, label in labels:
        cells = [label]
        for targets in TARGETS:
            for mode in ('deterministic', 'stochastic'):
                values = [r['coverage_rate'] for r in rows if r['targets'] == targets and r['mode'] == mode
                          and r['run'].rsplit('_', 2)[0] == algo]
                cells.append(f'{statistics.mean(values):.3f}' if values else '--')
        lines.append(' & '.join(cells) + ' \\\\')
    lines.append('\\midrule')
    for variant, label in (('greedy', 'MATE greedy cameras'), ('foresight', 'Option-level'), ('traj-2', 'Trajectories')):
        cells = [label]
        for targets in TARGETS:
            row = planners.get((targets, variant))
            cells += [f'{row["coverage"]:.3f}' if row else todo('running'), '--']
        lines.append(' & '.join(cells) + ' \\\\')
    write('v2_table_marl.tex', '\n'.join(lines) + '\n')
    for targets in TARGETS:
        for algo, name in (('mappo', 'Mappo'), ('ippo', 'Ippo'), ('hrl', 'Hrl')):
            values = [r['coverage_rate'] for r in rows if r['targets'] == targets and r['mode'] == 'deterministic'
                      and r['run'].rsplit('_', 2)[0] == algo]
            M.get(f'Marl{name}{SHORT[targets]}', statistics.mean(values) if values else None)


# Overhead, energy and latency ------------------------------------------------------------------ #


def overhead_table():
    main = rows_by(load('paper/main_4v8.json'))
    row_o, row_t = main.get(('greedy', 'foresight')), main.get(('greedy', 'traj-2'))
    entries = []
    if row_o:
        entries.append(('Option-level packet (pose, option intent, detections)', message_bytes(row_o, 'option')))
    if row_t:
        detections = DETECTION_FIXED_BYTES + DETECTION_BYTES_PER_TARGET * float(np.mean(row_t['per_camera']['camera_coverage'])) * 8
        entries.append(('Trajectories detection packet', detections))
        entries.append(('Trajectories trajectory intent (per publication)', float(INTENT_BYTES)))
        entries.append(('Trajectories, published every step (packet + intent)', detections + INTENT_BYTES))
        entries.append(('Trajectories, 2 synchronous rounds (packet + 2 intents)', detections + 2 * INTENT_BYTES))
    entries.append(('Belief maps of all targets (float16)', float(MAP_SHARING_BYTES)))
    entries.append(('Raw MATE camera observation (float32)', float(OBSERVATION_SHARING_BYTES)))
    lines = [f'{label} & {b:,.0f} & {energy_mj(b):.1f} \\\\'.replace(',', '\\,') for label, b in entries]
    write('v2_table_overhead.tex', '\n'.join(lines) + '\n')


def bench_table():
    rows = load('paper/bench_4v8.json')
    if not rows:
        write('v2_table_bench.tex', '\\multicolumn{3}{c}{' + todo('latency benchmark pending') + '} \\\\\n')
        return
    labels = {'greedy': 'MATE greedy cameras', 'foresight': 'Option-level planning', 'traj-1': 'Trajectories, 1 round',
              'traj-2': 'Trajectories, 2 rounds', 'traj-3': 'Trajectories, 3 rounds'}
    lines = []
    for variant, label in labels.items():
        one = next((r for r in rows if r['variant'] == variant and r['batch'] == 1), None)
        many = next((r for r in rows if r['variant'] == variant and r['batch'] > 1), None)
        rate = f'{many["env_steps_per_second"]:,.0f}'.replace(',', '\\,') if many else '--'
        lines.append(f'{label} & {one["ms_per_step"]:.1f} & {rate} \\\\'
                     if one and many else f'{label} & -- & -- \\\\')
        names = {'greedy': 'Greedy', 'foresight': 'O', 'traj-1': 'TOne', 'traj-2': 'TTwo', 'traj-3': 'TThree'}
        if one:
            M.get(f'Latency{names[variant]}', one['ms_per_step'], digits=1)
    write('v2_table_bench.tex', '\n'.join(lines) + '\n')


# Unified comparison with the MARL baselines ------------------------------------------------------ #

MARL_LABELS = [('mappo', 'MAPPO'), ('ippo', 'IPPO'), ('qmix', 'QMIX'), ('qplex', 'QPLEX'), ('riskq', 'RiskQ'),
               ('qfree', 'QFree'), ('macpf', 'MACPF')]


def _marl_rows():
    rows = load('paper/learned_4v8_final.json') or load('paper/learned_4v8_long.json') or load('paper/learned_4v8.json')
    grouped = {}
    for r in rows:
        grouped.setdefault((r['targets'], r['variant']), []).append(r)
    return grouped


def _network_cells(rows, kind):
    if not rows:
        return ['--'] * 5
    if 'per_camera' not in rows[0]:  # coverage-only evaluations
        return ['--'] * 4 + (['0'] if kind == 'none' else ['--'])
    pcs = [r['per_camera'] for r in rows]
    unique = np.mean([np.sum(pc['camera_unique_coverage']) for pc in pcs])
    redundancy = np.mean([r['redundancy'] for r in rows])
    fairness = np.mean([jain(pc['camera_coverage']) for pc in pcs])
    effort = np.mean([np.mean(pc['camera_rotation']) + np.mean(pc['camera_zoom']) for pc in pcs])
    size = '--' if kind is None else '0' if kind == 'none' else f'{message_bytes(rows[0], kind):.0f}'
    return [f'{unique:.2f}', f'{redundancy:.2f}', f'{fairness:.2f}', f'{effort:.2f}', size]


def unified_table():
    main = rows_by(load('paper/main_4v8.json'))
    bound = rows_by(load('paper/bound_4v8.json'), 'mode')
    marl = _marl_rows()
    lines = []

    def coverage_cell(rows):
        if not rows:
            return todo('running')
        values = [r['coverage'] for r in rows]
        if len(values) == 1:
            return f'{values[0]:.3f}'
        return f'{np.mean(values):.3f} $\\pm$ {np.std(values):.3f}'

    def add(label, by_target, kind, bold=False):
        cells = [label] + [coverage_cell(by_target.get(t)) for t in TARGETS] + _network_cells(by_target.get('greedy'), kind)
        if bold:
            cells[1:3] = [f'\\textbf{{{c}}}' for c in cells[1:3]]
        lines.append(' & '.join(cells) + ' \\\\')

    lines.append('\\multicolumn{8}{l}{\\emph{Rule-based}} \\\\')
    add('MATE greedy cameras', {t: [main[(t, 'greedy')]] for t in TARGETS if (t, 'greedy') in main}, 'option')
    lines.append('\\multicolumn{8}{l}{\\emph{Multi-agent reinforcement learning (raw pan/zoom commands, 3 seeds)}} \\\\')
    for algo, label in MARL_LABELS:
        add(label, {t: marl.get((t, algo)) for t in TARGETS if marl.get((t, algo))}, 'none')
    hitmac = {}
    for r in load('paper/hitmac_metrics.json'):
        hitmac.setdefault(r['targets'], []).append(r)
    if hitmac:
        lines.append('\\multicolumn{8}{l}{\\emph{Hierarchical MARL (scripted executors, $3\\times10^7$ steps, 3 seeds)}} \\\\')
        add('HiT-MAC', hitmac, None)
    for t in TARGETS:
        M.get(f'HitMac{SHORT[t]}', float(np.mean([r['coverage'] for r in hitmac[t]])) if t in hitmac else None)
    amortized = {}
    for r in load('paper/amortized_4v8.json'):
        if 'targets' in r:
            amortized.setdefault((r['targets'], r['variant']), []).append(r)
    if amortized:
        seeds = len(amortized.get(('greedy', 'amortized-student'), []))
        lines.append(f'\\multicolumn{{8}}{{l}}{{\\emph{{Learned by imitating \\FS{{}} (DAgger, {seeds} seed{"s" if seeds > 1 else ""})}}}} \\\\')
        add('Learned \\FS', {t: amortized[(t, 'amortized-student')] for t in TARGETS if (t, 'amortized-student') in amortized},
            'option')
    for t in TARGETS:
        rows = amortized.get((t, 'amortized-student'))
        M.get(f'Amortized{SHORT[t]}', float(np.mean([r['coverage'] for r in rows])) if rows else None)
        M.get(f'AmortizedSd{SHORT[t]}', float(np.std([r['coverage'] for r in rows])) if rows else None)
        rows = amortized.get((t, 'amortized-teacher'))
        M.get(f'AmortizedTeacher{SHORT[t]}', float(np.mean([r['coverage'] for r in rows])) if rows else None)
    for t in TARGETS:  # DAgger learning curves: initial policy and the mean of the last five evaluations
        starts, ends = [], []
        for path in sorted(glob.glob(os.path.join(RESULTS, 'amortized', f'{t}_s*.jsonl'))):
            with open(path, encoding='UTF-8') as file:
                rows = [json.loads(line) for line in file if line.strip()]
            curve = [r['eval_coverage'] for r in rows if 'eval_coverage' in r]
            start = next((r['reference']['initial_student'] for r in rows if 'reference' in r), None)
            if curve and start is not None:
                starts.append(start)
                ends.append(np.mean(curve[-5:]))
        M.get(f'AmortizedInit{SHORT[t]}', float(np.mean(starts)) if starts else None, digits=2)
        M.get(f'AmortizedCurve{SHORT[t]}', float(np.mean(ends)) if ends else None, digits=2)
    lines.append('\\multicolumn{8}{l}{\\emph{Model-based}} \\\\')
    add('Option-level planning', {t: [main[(t, 'foresight')]] for t in TARGETS if (t, 'foresight') in main}, 'option')
    add('FORESIGHT without learned forecaster', {t: [main[(t, 'traj-2')]] for t in TARGETS if (t, 'traj-2') in main}, 'traj2')
    add('FORESIGHT', {t: [main[(t, 'traj-2-forecast')]] for t in TARGETS if (t, 'traj-2-forecast') in main}, 'traj2', bold=True)
    lines.append('\\midrule')
    lines.append('Hindsight bound (clairvoyant) & ' + ' & '.join(cov(bound.get((t, 'clairvoyant-dp'))) for t in TARGETS)
                 + ' & -- & -- & -- & -- & -- \\\\')
    write('v2_table_unified.tex', '\n'.join(lines) + '\n')

    for t in TARGETS:
        best, best_value = None, -1.0
        for algo, label in MARL_LABELS:
            rows = marl.get((t, algo))
            if rows and np.mean([r['coverage'] for r in rows]) > best_value:
                best, best_value = label, float(np.mean([r['coverage'] for r in rows]))
        M.get(f'BestMarl{SHORT[t]}', best_value if best else None)
        M[f'BestMarlName{SHORT[t]}'] = best or todo('best MARL')
        fc = main.get((t, 'traj-2-forecast'))
        M.get(f'GainOverMarl{SHORT[t]}', (fc['coverage'] - best_value) if (fc and best) else None)


def ablation_table():
    main = rows_by(load('paper/main_4v8.json'))
    pub = rows_by(load('paper/published_4v8.json'))
    entries = [
        ('Option-level planning (one goal pose)', main, 'foresight'),
        ('Trajectories, 1 round', main, 'traj-1'),
        ('Trajectories, 2 rounds (reference)', main, 'traj-2'),
        ('Trajectories, 3 rounds', main, 'traj-3'),
        ('Published intents, every step', pub, 'pub-always'),
        ('Published intents, every 4th step', pub, 'pub-period-4'),
        ('Published intents, on change', pub, 'pub-event-1'),
        ('Hybrid memory', main, 'traj-2-hybrid'),
        ('No communication', main, 'traj-2-none'),
        ('+ learned forecaster', main, 'traj-2-forecast'),
    ]
    lines = []
    for label, source, variant in entries:
        cells = [label]
        for t in TARGETS:
            row, ref = source.get((t, variant)), (pub if source is pub else main).get((t, 'traj-2'))
            if row is None:
                cells += [todo('running'), '--']
                continue
            d, se, _ = paired(row, ref)
            cells += [f'{row["coverage"]:.3f}', '--' if variant == 'traj-2' else f'{d:+.3f} $\\pm$ {se:.3f}']
        lines.append(' & '.join(cells) + ' \\\\')
    write('v2_table_ablation.tex', '\n'.join(lines) + '\n')


def marl_curves():
    """Learning curves: (left) MARL baselines over 1e8 environment steps, (right) learned FORESIGHT (DAgger).

    Every curve starts with the evaluation of its initial policy at step 0 (research/eval_init.py)."""

    initial = {}
    path = os.path.join(RESULTS, 'paper', 'initial_coverage.json')
    if os.path.exists(path):
        with open(path, encoding='UTF-8') as file:
            initial = json.load(file)
    runs = {}
    curves_ppo = 'curves_final' if glob.glob(os.path.join(RESULTS, 'curves_final', '*.jsonl')) else 'curves_long'
    for folder in (curves_ppo, 'curves_value_long'):
        for path in sorted(glob.glob(os.path.join(RESULTS, folder, '*.jsonl'))):
            name = os.path.basename(path)[:-6]
            algo, targets, _ = name.rsplit('_', 2)
            with open(path, encoding='UTF-8') as file:
                rows = [json.loads(line) for line in file if line.strip()]
            points = [(r['env_steps'], r['eval_coverage']) for r in rows if 'eval_coverage' in r]
            if name in initial:
                points = [(0, initial[name])] + points
            if points:
                runs.setdefault((algo, targets), []).append(points)
    if not runs:
        return
    main = rows_by(load('paper/main_4v8.json'))
    bound = rows_by(load('paper/bound_4v8.json'), 'mode')
    style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharey='row', gridspec_kw={'width_ratios': [2.2, 1]})
    colors = plt.get_cmap('tab10')

    def references(ax, t):
        for variant, ls, lw in (('traj-2-forecast', '-', 1.2), ('foresight', '--', 0.9), ('greedy', ':', 0.9)):
            row = main.get((t, variant))
            if row:
                ax.axhline(row['coverage'], color='black', ls=ls, lw=lw)
        b = bound.get((t, 'clairvoyant-dp'))
        if b:
            ax.axhline(b['coverage'], color='#b22222', ls='-.', lw=0.9)

    for row_index, t in enumerate(TARGETS):
        ax = axes[row_index, 0]
        for index, (algo, label) in enumerate(MARL_LABELS):
            series = runs.get((algo, t))
            if not series:
                continue
            length = min(len(p) for p in series)
            x = np.asarray([p[0] for p in series[0][:length]]) / 1e6
            y = np.asarray([[v for _, v in p[:length]] for p in series])
            ax.plot(x, y.mean(0), color=colors(index), lw=1.3, label=label)
            ax.fill_between(x, y.mean(0) - y.std(0), y.mean(0) + y.std(0), color=colors(index), alpha=0.12, lw=0)
        references(ax, t)
        ax.set_xlim(0, 100)
        ax.set_ylabel(f'coverage rate\n({TARGET_TITLES[t].lower()})')
        ax = axes[row_index, 1]
        series = []
        for path in sorted(glob.glob(os.path.join(RESULTS, 'amortized', f'{t}_s*.jsonl'))):
            with open(path, encoding='UTF-8') as file:  # DAgger: 16 episodes x 500 steps per iteration
                rows = [json.loads(line) for line in file if line.strip()]
            start = next((r['reference']['initial_student'] for r in rows if 'reference' in r), None)
            points = ([(0.0, start)] if start is not None else []) + [
                ((r['iteration'] + 1) * 16 * 500 / 1e5, r['eval_coverage']) for r in rows if 'eval_coverage' in r]
            if len(points) > 1:
                series.append(points)
        if series:
            series = [p for p in series if len(p) == max(len(q) for q in series)]  # finished runs only
            length = min(len(p) for p in series)
            x = np.asarray([p[0] for p in series[0][:length]])
            y = np.asarray([[v for _, v in p[:length]] for p in series])
            ax.plot(x, y.mean(0), color='black', lw=1.4, marker='o', ms=2.2, label='Learned FORESIGHT (DAgger)')
            ax.fill_between(x, y.mean(0) - y.std(0), y.mean(0) + y.std(0), color='black', alpha=0.12, lw=0)
        references(ax, t)
        ax.set_xlim(0, 3.2)
    axes[0, 0].set_title('MARL baselines', fontsize=8)
    axes[0, 1].set_title('Learned FORESIGHT (DAgger)', fontsize=8)
    axes[1, 0].set_xlabel('environment steps ($\\times 10^6$)')
    axes[1, 1].set_xlabel('environment steps ($\\times 10^5$)')
    from matplotlib.lines import Line2D  # pylint: disable=import-outside-toplevel
    handles, _ = axes[0, 0].get_legend_handles_labels()
    handles += [Line2D([], [], color='black', lw=1.4, marker='o', ms=2.2, label='Learned FORESIGHT (DAgger)'),
                Line2D([], [], color='black', lw=1.2, label='FORESIGHT'),
                Line2D([], [], color='black', ls='--', lw=0.9, label='Option-level planning'),
                Line2D([], [], color='black', ls=':', lw=0.9, label='MATE greedy cameras'),
                Line2D([], [], color='#b22222', ls='-.', lw=0.9, label='Hindsight bound')]
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(os.path.join(FIGURES, 'v2_marl_curves.pdf'))
    plt.close(fig)


# Network-level evaluation: traffic trade-off, bursty / non-uniform loss, latency, scale ------------- #

TARGET_TITLES = {'greedy': 'Goal-directed targets', 'waypoint': 'Random Waypoint targets'}


def _bytes(row):
    return float(np.mean(row['per_camera']['camera_bytes'])) if row and 'camera_bytes' in row['per_camera'] else None


def _placeholders(*names):
    """TODO macros, overwritten by the results that exist."""

    for name in names:
        for tag in SHORT.values():
            M.get(f'{name}{tag}', None)


def tradeoff_figure():
    _placeholders('TradeFullBytes', 'TradeFullCov', 'TradeCheapBytes', 'TradeCheapCov', 'TradeCheapName', 'TradeSaving',
                  'TradeNoneCov', 'TradeMinBytes', 'TradeMinCov', 'TradeOptionBytes')
    rows = rows_by(load('paper/tradeoff_4v8.json'))
    main = rows_by(load('paper/main_4v8.json'))
    if not rows:
        return
    reports, periods = (1, 2, 4, 8), (1, 4, 16)
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    colors = {1: '#1f6fb2', 2: '#2a9d8f', 4: '#e08b2f', 8: '#b22222'}
    markers = {1: 'o', 4: 's', 16: '^'}
    for ax, targets in zip(axes, TARGETS):
        points = []
        for r in reports:
            for p in periods:
                row = rows.get((targets, f'net-r{r}-p{p}'))
                if row is None:
                    continue
                b = _bytes(row)
                points.append((b, row['coverage'], r, p))
                ax.scatter(b, row['coverage'], color=colors[r], marker=markers[p], s=22, zorder=3)
        none = rows.get((targets, 'net-none')) or main.get((targets, 'traj-2-none'))
        if none:
            points.append((0.0, none['coverage'], 0, 0))
            ax.scatter(0.3, none['coverage'], color='black', marker='x', s=22, zorder=3)
        front, best = [], -1.0
        for b, c, _, _ in sorted(points):
            if c > best:
                front.append((max(b, 0.3), c))
                best = c
        ax.plot(*zip(*front), color='gray', lw=0.9, ls='--', zorder=2, label='Pareto front')
        for variant, label, kind, ls in (('foresight', 'Option-level planning', 'option', ':'),
                                         ('greedy', 'MATE greedy cameras', 'option', '-.')):
            row = main.get((targets, variant))
            if row:
                ax.scatter(message_bytes(row, kind), row['coverage'], color='gray', marker='D', s=18, zorder=3)
                ax.annotate(label, (message_bytes(row, kind), row['coverage']), fontsize=6, ha='right',
                            xytext=(-5, -2), textcoords='offset points', color='dimgray')
        ax.set_xscale('log')
        ax.set_xlabel('bytes per node and step (log scale)')
        top = ax.secondary_xaxis('top', functions=(lambda b: energy_mj(b), lambda e: e / max(energy_mj(1.0), 1e-12)))
        top.set_xlabel('radio energy per node and step (mJ)', fontsize=7)
        top.tick_params(labelsize=6)
        ax.set_title(TARGET_TITLES[targets], fontsize=8, pad=26)
        # Macros: full traffic vs the cheapest configuration within 0.01 coverage of it.
        full = rows.get((targets, 'net-r1-p1'))
        if full:
            cheap = min((pt for pt in points if pt[2] and pt[1] >= full['coverage'] - 0.01), key=lambda pt: pt[0])
            tag = SHORT[targets]
            M.get(f'TradeFullBytes{tag}', _bytes(full), digits=1)
            M.get(f'TradeFullCov{tag}', full['coverage'])
            M.get(f'TradeCheapBytes{tag}', cheap[0], digits=1)
            M.get(f'TradeCheapCov{tag}', cheap[1])
            M[f'TradeCheapName{tag}'] = f'$R={cheap[2]}$, $P={cheap[3]}$'
            M.get(f'TradeSaving{tag}', 100.0 * (1.0 - cheap[0] / _bytes(full)), digits=0)
            if none:
                M.get(f'TradeNoneCov{tag}', none['coverage'])
            cheapest = min((pt for pt in points if pt[2]), key=lambda pt: pt[0])
            M.get(f'TradeMinBytes{tag}', cheapest[0], digits=1)
            M.get(f'TradeMinCov{tag}', cheapest[1])
            option = main.get((targets, 'foresight'))
            M.get(f'TradeOptionBytes{tag}', message_bytes(option, 'option') if option else None, digits=0)
    axes[0].set_ylabel('coverage rate')
    from matplotlib.lines import Line2D  # pylint: disable=import-outside-toplevel
    handles = [Line2D([], [], color=colors[r], marker='o', ls='', ms=4, label='packets every step' if r == 1 else f'packets every {r} steps')
               for r in reports]
    handles += [Line2D([], [], color='black', marker=markers[p], ls='', ms=4, fillstyle='none',
                       label='intents every step' if p == 1 else f'intents every {p} steps') for p in periods]
    handles += [Line2D([], [], color='black', marker='x', ls='', ms=4, label='no communication'),
                Line2D([], [], color='gray', ls='--', lw=0.9, label='Pareto front')]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(os.path.join(FIGURES, 'v2_tradeoff.pdf'))
    plt.close(fig)


def burst_figure():
    _placeholders('BurstIid', 'BurstFive', 'BurstTwenty', 'BurstSpread', 'BurstPubHalf', 'BurstPub', 'BurstTwentyHalf',
                  'BurstIidHalf')
    rows = rows_by(load('paper/burst_4v8.json'))
    main = rows_by(load('paper/main_4v8.json'))
    if not rows:
        return
    losses = (0.0, 0.25, 0.5, 0.75, 0.9)
    series = [('iid-{p}', 'i.i.d. loss', '#1f6fb2', '-'), ('burst5-{p}', 'bursts of 5 steps', '#e08b2f', '-'),
              ('burst20-{p}', 'bursts of 20 steps', '#b22222', '-'), ('spread-{p}', 'non-uniform ($\\pm$80%)', '#2a9d8f', '--'),
              ('pub-burst20-{p}', 'published intents, bursts of 20, intents lost too', '#6a3d9a', ':')]
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, targets in zip(axes, TARGETS):
        zero = rows.get((targets, 'traj-2-hybrid'))
        for pattern, label, color, ls in series:
            values = [zero['coverage'] if (p == 0.0 and zero) else
                      (rows[(targets, pattern.format(p=p))]['coverage'] if (targets, pattern.format(p=p)) in rows else np.nan)
                      for p in losses]
            if pattern.startswith('pub'):
                values[0] = np.nan
            ax.plot(losses, values, marker='o', ms=3, lw=1.4, color=color, ls=ls, label=label)
        for variant, label, ls in (('traj-2-none', 'no communication', ':'), ('foresight', 'option-level, ideal channel', '-.')):
            row = main.get((targets, variant))
            if row:
                ax.axhline(row['coverage'], color='gray', ls=ls, lw=0.9, label=label)
        ax.set_title(TARGET_TITLES[targets])
        ax.set_xlabel('mean packet loss rate $p$')
        tag = SHORT[targets]
        for key, macro in (('iid-0.9', 'BurstIid'), ('burst5-0.9', 'BurstFive'), ('burst20-0.9', 'BurstTwenty'),
                           ('spread-0.9', 'BurstSpread'), ('pub-burst20-0.5', 'BurstPubHalf'),
                           ('pub-burst20-0.9', 'BurstPub'), ('burst20-0.5', 'BurstTwentyHalf'), ('iid-0.5', 'BurstIidHalf')):
            row = rows.get((targets, key))
            M.get(f'{macro}{tag}', row['coverage'] if row else None)
    axes[0].set_ylabel('coverage rate')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(os.path.join(FIGURES, 'v2_burst.pdf'))
    plt.close(fig)


def delay_figure():
    _placeholders(*(f'Delay{k}{n}' for k in ('Hybrid', 'Team', 'Intent') for n in ('Zero', 'Ten', 'Twenty')))
    rows = rows_by(load('paper/delay_4v8.json'))
    main = rows_by(load('paper/main_4v8.json'))
    if not rows:
        return
    delays = (0, 1, 2, 5, 10, 20)
    series = [('delay-hybrid-{d}', 'detections delayed, hybrid memory', '#1f6fb2', '-', 'traj-2-hybrid'),
              ('delay-team-{d}', 'detections delayed, shared memory', '#e08b2f', '--', None),
              ('delay-intent-{d}', 'intents delayed (published every 4th step)', '#6a3d9a', '-', 'delay-intent-0')]
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, targets in zip(axes, TARGETS):
        for pattern, label, color, ls, zero in series:
            values = []
            for d in delays:
                if d == 0:
                    row = rows.get((targets, zero)) if zero else main.get((targets, 'traj-2'))
                else:
                    row = rows.get((targets, pattern.format(d=d)))
                values.append(row['coverage'] if row else np.nan)
            ax.plot(delays, values, marker='o', ms=3, lw=1.4, color=color, ls=ls, label=label)
            key = pattern.split('-')[1]
            for d, name in ((10, 'Ten'), (20, 'Twenty')):
                M.get(f'Delay{key.capitalize()}{name}{SHORT[targets]}', values[delays.index(d)])
            M.get(f'Delay{key.capitalize()}Zero{SHORT[targets]}', values[0])
        row = main.get((targets, 'traj-2-none'))
        if row:
            ax.axhline(row['coverage'], color='gray', ls=':', lw=0.9, label='no communication')
        ax.set_title(TARGET_TITLES[targets])
        ax.set_xlabel('message latency (steps)')
    axes[0].set_ylabel('coverage rate')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(os.path.join(FIGURES, 'v2_delay.pdf'))
    plt.close(fig)


SCALE_CONFIGS = ('4v8', '8v8', '8v16', '16v32')
SCALE_TAGS = {'4v8': '', '8v8': 'Eight', '8v16': 'EightSixteen', '16v32': 'Sixteen'}


def scale_results():
    data = {}
    for config in SCALE_CONFIGS:
        rows = rows_by(load(f'paper/main_{config}.json'))
        if config == '4v8':
            rows.update({k: v for k, v in rows_by(load('paper/published_4v8.json')).items() if k[1] == 'pub-period-4'})
        if config in ('4v8', '8v8'):
            traj = rows_by(load(f'paper/headroom_traj2_{config}.json'), 'mode')
            for t in TARGETS:
                if (t, 'traj-2') not in rows and (t, 'foresight') in traj:
                    rows[(t, 'traj-2')] = traj[(t, 'foresight')]
        for key, row in rows_by(load(f'paper/main_{config}_extra.json')).items():
            if key[1] in ('pub-period-4', 'traj-2-none'):  # variants added after the first 8v8 run
                rows[key] = row
        learned = {}
        for r in load(f'paper/amortized_{config}.json'):
            if r.get('variant') == 'amortized-student':
                learned.setdefault(r['targets'], []).append(r['coverage'])
        for t, values in learned.items():
            rows[(t, 'learned')] = {'coverage': float(np.mean(values))}
        for t in TARGETS:
            row = rows.get((t, 'learned'))
            M.get(f'Learned{SHORT[t]}{SCALE_TAGS[config]}', row['coverage'] if row else None)
        bound = rows_by(load(f'paper/bound_{config}.json'), 'mode')
        for t in TARGETS:
            if (t, 'clairvoyant-dp') in bound:
                rows[(t, 'bound')] = bound[(t, 'clairvoyant-dp')]
        data[config] = rows
    labels = [('greedy', 'MATE greedy cameras'), ('foresight', 'Option-level planning'),
              ('traj-1', 'Trajectories, 1 round'), ('traj-2', 'Trajectories, 2 rounds'),
              ('pub-period-4', 'Published intents, every 4th step'), ('learned', 'Learned \\FS'),
              ('traj-2-none', 'No communication')]
    lines = []
    for variant, label in labels + [('bound', 'Hindsight bound')]:
        if variant == 'bound':
            lines.append('\\midrule')
        cells = [label]
        for config in SCALE_CONFIGS:
            for t in TARGETS:
                row = data[config].get((t, variant))
                cells.append(f'{row["coverage"]:.3f}' if row else '--')
        lines.append(' & '.join(cells) + ' \\\\')
    cells = ['Gap closed (2 rounds)']
    for config in SCALE_CONFIGS:
        for t in TARGETS:
            o, t2, b = (data[config].get((t, v)) for v in ('foresight', 'traj-2', 'bound'))
            closed = 100.0 * (t2['coverage'] - o['coverage']) / (b['coverage'] - o['coverage']) if (o and t2 and b) else None
            cells.append(f'{closed:.0f}\\%' if closed is not None else '--')
            tag = SCALE_TAGS[config]
            if config in ('8v16', '16v32'):
                M.get(f'ClosedT{SHORT[t]}{tag}', closed, digits=0)
                M.get(f'TTwo{SHORT[t]}{tag}', t2['coverage'] if t2 else None)
                M.get(f'O{SHORT[t]}{tag}', o['coverage'] if o else None)
                M.get(f'Bound{SHORT[t]}{tag}', b['coverage'] if b else None)
                g = data[config].get((t, 'greedy'))
                M.get(f'Greedy{SHORT[t]}{tag}', g['coverage'] if g else None)
    lines.append(' & '.join(cells) + ' \\\\')
    cells, gaps = ['Lagrangian bound (from above)'], []
    for config in SCALE_CONFIGS:
        dual = rows_by(load(f'paper/bound_dual_{config}.json'), 'mode')
        for t in TARGETS:
            row, b = dual.get((t, 'clairvoyant-dp')), data[config].get((t, 'bound'))
            cells.append(f'{row["dual_bound"]:.3f}' if row and 'dual_bound' in row else '--')
            if row and 'dual_bound' in row and b:
                gaps.append(row['dual_bound'] - b['coverage'])
            if config in ('8v16', '16v32'):
                M.get(f'Dual{SHORT[t]}{SCALE_TAGS[config]}', row.get('dual_bound') if row else None)
    lines.insert(len(lines) - 1, ' & '.join(cells) + ' \\\\')
    M.get('DualGapMax', max(gaps) if len(gaps) == 2 * len(SCALE_CONFIGS) else None)
    write('v2_table_scale.tex', '\n'.join(lines) + '\n')

    bench = {}
    for c in SCALE_CONFIGS:  # minimum over repeated runs (the GPU was shared with other jobs)
        rows = [r for name in (f'paper/bench_scale_{c}.json', f'paper/bench_scale_run1/bench_scale_{c}.json')
                for r in load(name) if r['batch'] == 1]
        bench[c] = {}
        for r in rows:
            if r['variant'] not in bench[c] or r['ms_per_step'] < bench[c][r['variant']]['ms_per_step']:
                bench[c][r['variant']] = r
    lines = []
    for variant, label in (('greedy', 'MATE greedy cameras'), ('foresight', 'Option-level planning'),
                           ('traj-2', 'Trajectories, 2 synchronous rounds'), ('pub-4', 'Published intents, shared memory'),
                           ('pub-4-hybrid', 'Published intents, hybrid memory')):
        cells = [label]
        for config in SCALE_CONFIGS:
            row = bench[config].get(variant)
            cells.append(f'{row["ms_per_step"]:.1f}' if row else '--')
        lines.append(' & '.join(cells) + ' \\\\')
    cells, planner_cells = ['Learned \\FS'], {}
    for config in SCALE_CONFIGS:  # trained weights; 4v8: minimum over the two measurements
        rows = [r for name in (f'paper/amortized_{config}.json', f'paper/amortized_{config}_seed0.json')
                for r in load(name) if r.get('variant') == 'latency' and r['config'] == f'MATE-{config}-9.yaml']
        student_ms = min((r['student_ms'] for r in rows), default=None)
        planner_ms = min((r['planner_ms'] for r in rows), default=None)
        cells.append(f'{student_ms:.1f}' if student_ms is not None else '--')
        tag = {'4v8': 'Four', '8v8': 'Eight', '8v16': 'EightSixteen', '16v32': 'Sixteen'}[config]
        M.get(f'AmortizedMs{tag}', student_ms, digits=1)
        M.get(f'PlannerMs{tag}', planner_ms, digits=1)
    lines.append(' & '.join(cells) + ' \\\\')
    write('v2_table_bench_scale.tex', '\n'.join(lines) + '\n')
    for config in ('16v32',):
        for variant, name in (('traj-2', 'TTwo'), ('pub-4', 'Pub'), ('pub-4-hybrid', 'PubHybrid')):
            row = bench[config].get(variant)
            M.get(f'Latency{name}Sixteen', row['ms_per_step'] if row else None, digits=1)


def main():
    os.makedirs(FIGURES, exist_ok=True)
    gap_table()
    main_table()
    scale_results()
    tradeoff_figure()
    burst_figure()
    delay_figure()
    loss_results()
    published_table()
    comm_tables()
    comm_curves()
    learning_table()
    marl_table()
    overhead_table()
    bench_table()
    unified_table()
    ablation_table()
    marl_curves()
    M.write()
    print('wrote', sorted(f for f in os.listdir(GENERATED) if f.startswith('v2_')))


if __name__ == '__main__':
    main()
