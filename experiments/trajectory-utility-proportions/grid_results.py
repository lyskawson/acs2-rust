import argparse
from collections import defaultdict
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
from statistics import mean, median, stdev

from analysis_rules import carry_share
from cost_forecast import json_lines, write_csv
from grid_plan import BINARY_SHA256, MEASUREMENT_COMMIT, derive
from grid_protocol import T19, load_plan
from proportions import ABLATION, HERE, MEAN, RANDOM, S1, SHARES, checksum, label, read, registered
from agent_names import canonical_run_name
from recovery_plan import (FIRST_MANIFEST, FIRST_PLAN, ORIGINAL_COMPLETE, PREFIX_COUNT, RETRY_COUNT,
                           derive_recovery, sealed_stop)
from register_protocol import exact_floor
from validate_proportions import capped_steps, close

ARCHIVE_INDEX = '9693019dfdf93a3c42d8198f29c8a3e216ae216834359ea4a8145b46542ecc2d'
EASY_LABELS = '141deaf403b0ef3012d2e130217e3de4ab64390c5d1a2ac3d2ac1646edca47ce'
ABLATION_COSTS = {
    'grid-run-costs.csv': '70c7ec9469da29f82da43bc8232bde57de0f7d5384df3f484d76c62c80edf88b',
    'reference-run-costs.csv': 'e299621409ea0a2575ff1c1c74af38ef7e2e023bae8fb663985f60b1c109b2e5',
}
NEW_ARMS = tuple(label(agent, share) for agent in (MEAN, RANDOM) for share in SHARES)
REFERENCE_ARMS = (S1, label(MEAN, 0.5), label(RANDOM, 0.5))
GROUPS = ('no_aliasing', 'aliasing')
EASY_SOURCE = 'Original ACS2 at the registered cap (ablation)'


def verified(path, digest):
    if checksum(path) != digest:
        raise ValueError(f'input checksum differs: {path}')
    return path


def csv_rows(path, delimiter=','):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def test_references(configs):
    geometry = {row['task']: row for row in read(ABLATION / 'geometry.json')}
    result = {}
    for task, cfg in configs.items():
        entry = geometry[task]
        spec = cfg['registration']['blocks']['test']
        pairs = {tuple(row['goal']): [tuple(start) for start in row['starts']]
                 for row in spec['goal_starts']}
        index = {tuple(cell): i for i, cell in enumerate(entry['walkable_cells'])}
        count = sum(map(len, pairs.values()))
        if count != spec['evaluation_pairs'] or Fraction(spec['pair_weight']) != Fraction(1, count):
            raise ValueError('test pair counts or equal weights differ from registration')
        base = exact_floor(entry, pairs, cfg['cap'])
        if base != Fraction(spec['random_floor_exact']):
            raise ValueError('maze transitions do not reproduce the registered random floor')
        shortest = Fraction(sum(entry['distances'][index[start]][index[goal]]
                                for goal, starts in pairs.items() for start in starts), count)
        result[task] = dict(registered_floor=base, extended_floor=exact_floor(entry, pairs, 100),
                            shortest=float(shortest), pairs=count)
    return result


def entries(index, users, rename, costs, expected, source):
    result = {}
    for entry in read(index):
        if entry['agent'] not in users:
            continue
        key = entry['task'], rename(entry['agent']), entry['seed']
        if key in result or key not in costs:
            raise ValueError('absent or duplicate archive cost')
        result[key] = dict(path=Path(entry['source']), sha256=entry['sha256'], source=source, **costs[key])
    if result.keys() != expected:
        raise ValueError('the reference archive differs from the registered cells')
    return result


def old_sources(archive, measures, configs, seeds, final):
    simple_costs = {}
    for name, digest in ABLATION_COSTS.items():
        path = verified(archive / 'final-evidence/grid-evidence' / name, digest)
        for row in csv_rows(path):
            if row['agent'] != S1 or row.get('state', 'COMPLETED') != 'COMPLETED':
                continue
            key = row['task'], S1, int(row['seed'])
            if key in simple_costs:
                raise ValueError('duplicate S+1 cost')
            simple_costs[key] = dict(process=float(row['process_cpu_seconds_wait4']),
                                     peak=int(row['peak_rss_kib']), allocated=float(row['allocation_cpu_seconds']))
    simple = entries(verified(archive/'final-evidence/local-full-index.json', ARCHIVE_INDEX),
                     (S1,), lambda name: name, simple_costs,
                     {(task, S1, seed) for task in configs for seed in seeds}, 'ablation')
    archive_hashes = final['costs']['archive_sha256']
    measured_costs = {}
    for row in csv_rows(verified(measures/'registered-analysis/run-costs.csv', archive_hashes['run-costs'])):
        if row['arm'] not in (MEAN, RANDOM):
            continue
        key = row['task'], label(row['arm'], 0.5), int(row['seed'])
        if key in measured_costs:
            raise ValueError('duplicate 6a cost')
        measured_costs[key] = dict(process=float(row['process_cpu_seconds']), peak=int(row['peak_rss_kib']),
                                  allocated=float(row['measurement_step_or_allocation_cpu_seconds']))
    old = entries(verified(measures/'registered-analysis/full-index.json', archive_hashes['full-index']),
                  (MEAN, RANDOM), lambda agent: label(agent, 0.5), measured_costs,
                  {(task, arm, seed) for task in configs for arm in REFERENCE_ARMS[1:] for seed in seeds}, 'measures')
    return simple | old


def sealed_reader(root, manifest):
    def read_file(name):
        return load_plan(root / name, manifest[name])
    return read_file


def measurement_steps(root, manifest, partial=False):
    load = sealed_reader(root, manifest)
    prefix = 'partial-' if partial else ''
    allocations = load(f'{prefix}allocations.json')
    accounting = f'{prefix}accounting.psv'
    records = {row['JobIDRaw']: row for row in csv_rows(
        verified(root/accounting, manifest[accounting]), delimiter='|') if '.' in row['JobIDRaw']}
    raw_ids = {job: row['JobIDRaw' if partial else 'raw_id'] for job, row in allocations.items()}
    return records, raw_ids


def complete_source(run, cfg, root, manifest, steps, used_steps):
    load = sealed_reader(root, manifest)
    attempt = f"attempts/{run['run_id']}"
    folder = f"{attempt}/measurement/{cfg['id']}_{run['agent']}_s{run['seed']}"
    completion = load(f'{folder}/completion.json')
    step = load(f'{attempt}/step.json')
    job, raw_step = step['job_id'], f"{step['job_id_raw']}.{step['step_id']}"
    records, raw_ids = steps
    record = records[raw_step]
    charged = int(record['CPUTimeRAW'])
    if (raw_step in used_steps or raw_ids[job] != step['job_id_raw']
            or record['JobID'] != f"{job}.{step['step_id']}"
            or record['AllocCPUS'] != '1' or record['ExitCode'] != '0:0'
            or record['State'] != 'COMPLETED' or not 0 < charged <= step['limit_seconds']):
        raise ValueError('measurement step is not a distinct registered Slurm charge')
    used_steps.add(raw_step)
    reproduced = 0
    if reference := run.get('previous') or run['pilot']:
        recovery = run.get('previous') is not None
        filename = 'recovery-prefix-identity.json' if recovery else 'prefix-identity.json'
        field = 'previous_sha256' if recovery else 'pilot_sha256'
        proof = load(f'{folder}/{filename}')
        if proof != {'rows': reference['checkpoints'], 'differences': 0, field: reference['sha256']}:
            raise ValueError('preserved prefix proof does not match its source')
        reproduced = proof['rows']
    return completion, root / folder / 'rows.jsonl', manifest[f'{folder}/rows.jsonl'], charged, reproduced


def measured_sources(pilot, grid, grid_manifest_sha, final, recovery=None, recovery_sha=None):
    pilot_manifest = load_plan(pilot/'files.sha256.json', final['pilot_manifest_sha256'])
    pilot_summary = load_plan(pilot/'pilot-summary.json', pilot_manifest['pilot-summary.json'])
    data = derive(final, pilot_manifest, pilot_summary)
    if (recovery is None) != (recovery_sha is None):
        raise ValueError('a recovery needs its sealed manifest checksum')
    recovered, recovery_overlap = {}, 0
    if recovery is None:
        manifest = load_plan(grid / 'files.sha256.json', grid_manifest_sha)
        load = sealed_reader(grid, manifest)
        summary = load('accounting-summary.json')
        if (summary['stopped'] or summary['completed_runs'] != 3255 or summary['submitted_runs'] != 3255
                or summary['reserved_seconds'] or summary['reused'] != 105
                or load('operations.json')['plan_sha256'] != checksum(HERE / 'final-plan.json')):
            raise ValueError('registered analysis requires the fully completed grid')
    else:
        if grid_manifest_sha != FIRST_MANIFEST or checksum(HERE / 'final-plan.json') != FIRST_PLAN:
            raise ValueError('recovery requires the original sealed stopped grid')
        manifest, original, _ = sealed_stop(grid)
        recovery_manifest = load_plan(recovery / 'files.sha256.json', recovery_sha)
        load = sealed_reader(recovery, recovery_manifest)
        summary = load('accounting-summary.json')
        earlier_spent = original['grid_cpu_seconds']
        if (summary['stopped'] or summary['completed_runs'] != RETRY_COUNT
                or summary['submitted_runs'] != RETRY_COUNT or summary['reserved_seconds']
                or summary['reused'] != 105 or summary['earlier_completed_runs'] != ORIGINAL_COMPLETE
                or summary['earlier_grid_cpu_seconds'] != earlier_spent
                or summary['reproduced_checkpoints'] != PREFIX_COUNT
                or summary['grid_total_pilot_and_spent_cpu_seconds'] != original['total_cpu_seconds'] + summary['grid_cpu_seconds']
                or load('operations.json')['plan_sha256'] != FIRST_PLAN):
            raise ValueError('the recovered grid is incomplete or its accounting differs')
        approved = load('operations/experiments/trajectory-utility-proportions/recovery-registration.json')
        scheduled = derive_recovery(data, grid, approved)
        recovered = {run['run_id']: run for run in scheduled['runs']}
        recovered_steps = measurement_steps(recovery, recovery_manifest)
    steps = measurement_steps(grid, manifest, partial=recovery is not None)
    result, used_steps, overlap = {}, set(), 0
    for run in data['reused'] + data['runs']:
        cfg = data['configs'][run['task']]
        key = run['task'], run['arm'], run['seed']
        if 'phase' not in run:
            reference = run['pilot']
            path = pilot / reference['path']
            relative = str(path.with_name('completion.json').relative_to(pilot))
            completion = load_plan(pilot / relative, pilot_manifest[relative])
            digest, charged, source = reference['sha256'], reference['slurm_cpu_seconds'], 'pilot'
        elif retry := recovered.get(run['run_id']):
            completion, path, digest, charged, reproduced = complete_source(
                retry, cfg, recovery, recovery_manifest, recovered_steps, used_steps)
            recovery_overlap += reproduced
            source = 'recovery'
        else:
            completion, path, digest, charged, reproduced = complete_source(
                run, cfg, grid, manifest, steps, used_steps)
            overlap += reproduced
            source = 'grid'
        if completion['sha256']['rows.jsonl'] != digest or completion['rows_seen'] != len(cfg['targets']):
            raise ValueError('measurement completion differs from the sealed rows')
        result[key] = dict(path=path, sha256=digest, source=source, allocated=charged,
                           process=completion['user_cpu_seconds']+completion['system_cpu_seconds'],
                           peak=completion['peak_rss_kib'])
    if len(result) != 3360 or len(used_steps) != 3255 or recovery is not None and recovery_overlap != PREFIX_COUNT:
        raise ValueError('the measurement archives do not cover the whole grid')
    return result, summary, overlap


def raw_inputs(sources, configs, references, names):
    observations, carry, run_costs = defaultdict(list), {}, []
    for (task, arm, seed), source in sorted(sources.items()):
        cfg = configs[task]
        registered_cap = cfg['cap']
        reference = references[task]
        host_measurement = source['source'] in ('pilot', 'grid', 'recovery')
        rows = json_lines(verified(source['path'], source['sha256']))
        if [row['nominal_step'] for row in rows] != cfg['targets']:
            raise ValueError('checkpoint horizons differ from registration')
        validation = []
        for row in rows:
            agent = S1 if arm == S1 else arm.rsplit('_p', 1)[0]
            if ((row['task'], canonical_run_name(row['agent']), row['seed'], row['cap']) != (task, agent, seed, registered_cap)
                    or host_measurement and (row['commit'] != MEASUREMENT_COMMIT or row['source_state'] != 'clean')):
                raise ValueError('measurement row identity differs')
            if host_measurement and row['agent_parameters']['trajectory_utility']['tu_share'] not in SHARES:
                raise ValueError('measurement row share is outside the grid')
            if host_measurement and arm != label(agent, row['agent_parameters']['trajectory_utility']['tu_share']):
                raise ValueError('measurement row share differs from the run label')
            blocks = row['evaluation_blocks']
            test = blocks['test']
            if (test['evaluation_pairs'] != reference['pairs']
                    or ('test_caps' in row) != host_measurement):
                raise ValueError('test caps or pairs differ from registration')
            short = capped_steps(test['success'], test['mean_success_steps'], registered_cap)
            caps = [('registered', registered_cap, test['success'], short)]
            if host_measurement:
                values = row['test_caps']
                close(values['registered']['mean_steps'], short)
                close(values['registered']['mean_shortest_path'], reference['shortest'])
                close(values['extended']['mean_shortest_path'], reference['shortest'])
                caps.append(('extended', 100, values['extended']['success'], values['extended']['mean_steps']))
            utility = row.get('trajectory_utility', {})
            requested = row['agent_parameters'].get('trajectory_utility', {}).get('route_probabilities', {})
            realized = utility.get('diagnostics', {}).get('realized_copy_shares', {})
            for test_cap, steps, success, mean_steps in caps:
                if not 0 <= success <= 1+1e-12:
                    raise ValueError('invalid test success')
                observations[task, arm, test_cap].append(dict(
                    task=task, maze=names[task], group=cfg['group'], arm=arm, test_cap=test_cap,
                    cap_steps=steps, seed=seed, nominal_step=row['nominal_step'], success=success,
                    mean_steps=mean_steps, mean_shortest_path=reference['shortest'],
                    mean_steps_over_shortest=mean_steps/reference['shortest'], knowledge=row['knowledge'],
                    requested_tu_share=requested.get('trajectory_utility', 0),
                    requested_hindsight_share=requested.get('hindsight', 2 / 3),
                    realized_tu_share=realized.get('trajectory_utility', 0),
                    realized_hindsight_share=realized.get('hindsight', row['replay_diagnostics']['realized_relabeled_share']),
                ))
            validation.append(dict(nominal_step=row['nominal_step'], evaluation_blocks={
                'validation': {'success': blocks['validation']['success']}}))
        carry[task, arm, seed] = validation
        run_costs.append(dict(task=task, maze=names[task], group=cfg['group'], arm=arm, seed=seed,
                              source=source['source'], process_cpu_seconds=source['process'],
                              utility_cpu_seconds=rows[-1].get('trajectory_utility', {}).get('cost', {}).get('cpu_seconds', 0),
                              peak_rss_kib=source['peak'], slurm_cpu_seconds=source['allocated'],
                              extended_test_cpu_seconds=rows[-1].get('test_caps', {}).get('extended', {}).get('cumulative_cpu_seconds'),
                              path=str(source['path']), sha256=source['sha256']))
    return observations, carry, run_costs


def interval(values):
    se = stdev(values) / math.sqrt(20)
    center = mean(values)
    return dict(n=20, mean=center, se=se, ci_low=center-T19*se, ci_high=center+T19*se)


def verdict(positive, negative, count):
    if not count:
        return 'not assessed'
    if positive and negative:
        return 'mixed'
    if positive >= (count + 1) // 2:
        return 'improves'
    if negative >= (count + 1) // 2:
        return 'worsens'
    return 'no clear effect'


def speed_points(points, ratios, benchmark):
    first = next((point for point, value in zip(points, ratios) if value >= .95), None)
    sustained = next((point for index, point in enumerate(points)
                      if all(value >= .95 for value in ratios[index:])), None)
    matched = next((point for point, value in zip(points, ratios) if value >= benchmark), None)
    return dict(first_095=first, sustained_095=sustained, reaches_s1_final=matched)


def statistics(study, configs, names, references, easy, observed, run_costs):
    seeds = study['seeds']
    results = defaultdict(list)
    for cap in ('registered', 'extended'):
        arms = NEW_ARMS + (REFERENCE_ARMS if cap == 'registered' else ())
        contrasts = [(label(a, share), S1 if b == S1 else label(b, share))
                     for share in SHARES for a, b in study['analysis']['contrasts']
                     if cap == 'registered' or b == RANDOM]
        for task, cfg in configs.items():
            common = dict(task=task, maze=names[task], group=cfg['group'], test_cap=cap,
                          cap_steps=cfg['cap'] if cap == 'registered' else 100)
            points = cfg['targets']
            scores, final, curves = {}, {}, {}
            for arm in arms:
                values = observed[task, arm, cap]
                by_seed = {seed: [row for row in values if row['seed']==seed] for seed in seeds}
                if any([row['nominal_step'] for row in r] != points for r in by_seed.values()):
                    raise ValueError('arm data lack paired seeds and checkpoints')
                for seed, rows in by_seed.items():
                    success = [row['success'] for row in rows]
                    scores[arm, seed] = dict(primary=mean(success), final=success[-1])
                    results['seed-scores'].append(dict(common, arm=arm, seed=seed, **scores[arm,seed]))
                for metric in ('primary', 'final'):
                    stats = interval([scores[arm, seed][metric] for seed in seeds])
                    results['arm-success'].append(dict(common, arm=arm, measure=metric, **stats))
                    if metric == 'final':
                        final[arm] = stats
                across = [mean(row['success'] for row in rows) for rows in zip(*by_seed.values())]
                curves[arm] = across
                for point, value, rows in zip(points, across, zip(*by_seed.values())):
                    results[f'curves-{cap}'].append(dict(common, arm=arm, nominal_step=point, n=20, mean=value,
                        mean_steps=mean(row['mean_steps'] for row in rows),
                        mean_shortest_path=references[task]['shortest'],
                        mean_steps_over_shortest=mean(row['mean_steps_over_shortest'] for row in rows),
                        requested_tu_share=rows[0]['requested_tu_share'],
                        requested_hindsight_share=rows[0]['requested_hindsight_share'],
                        realized_tu_share=mean(row['realized_tu_share'] for row in rows),
                        realized_hindsight_share=mean(row['realized_hindsight_share'] for row in rows)))
            if cap == 'registered':
                benchmark = final[S1]['mean']
                for arm, values in curves.items():
                    results['speed'].append(dict(common, arm=arm, s1_final=benchmark,
                                                 **speed_points(points, values, benchmark)))
            floor = references[task][f'{cap}_floor']
            compared = {arm for pair in contrasts for arm in pair}
            is_floor = max(Fraction(final[arm]['ci_low']) for arm in compared) <= floor
            flags = dict(floor=is_floor, too_easy=easy[task], too_easy_source=EASY_SOURCE)
            results['labels'].append(dict(common, **flags, random_floor_exact=str(floor), random_floor=float(floor)))
            for treatment, control in contrasts:
                ceiling = min(final[treatment]['mean'], final[control]['mean']) >= .99
                tag = f'{treatment} - {control}'
                for metric in ('primary', 'final'):
                    diffs = [scores[treatment, seed][metric]-scores[control, seed][metric] for seed in seeds]
                    results['comparisons'].append(dict(
                        common, contrast=tag, measure=metric, favorable=treatment, reference=control,
                        ceiling=ceiling, counted=not (is_floor or (metric=='primary' and easy[task])
                                                     or (metric=='final' and ceiling)),
                        **flags, **interval(diffs)))
                    for seed, diff in zip(seeds, diffs):
                        results['paired-differences'].append(dict(common, contrast=tag, measure=metric,
                                                                  seed=seed, difference=diff))
    keys = dict.fromkeys((r['test_cap'],r['contrast'],r['measure']) for r in results['comparisons'])
    for cap, contrast, measure in keys:
        rows = [r for r in results['comparisons'] if (r['test_cap'],r['contrast'],r['measure']) ==
                (cap,contrast,measure) and r['counted']]
        for group in (*GROUPS,'all'):
            group_rows = [r for r in rows if group in ('all',r['group'])]
            plus, minus = sum(r['ci_low']>0 for r in group_rows), sum(r['ci_high']<0 for r in group_rows)
            n = len(group_rows)
            results['grouped'].append(dict(test_cap=cap, group=group, contrast=contrast, measure=measure,
                                           A_plus=plus, A_minus=minus, n=n, too_easy_source=EASY_SOURCE,
                                           verdict=verdict(plus,minus,n) if group!='all' else None))
    costs = {(r['task'],r['arm'],r['seed']): r for r in run_costs}
    for key, row in costs.items():
        s1 = costs[key[0],S1,key[2]]['process_cpu_seconds']
        row['process_to_s1'] = row['process_cpu_seconds']/s1
        row['utility_to_s1'] = row['utility_cpu_seconds']/s1
    for task, cfg in configs.items():
        for arm in NEW_ARMS + REFERENCE_ARMS:
            rows = [costs[task,arm,seed] for seed in seeds]
            results['maze-costs'].append(dict(task=task, maze=names[task], group=cfg['group'], arm=arm, seeds=20,
                median_process_to_s1=median(r['process_to_s1'] for r in rows),
                median_utility_to_s1=median(r['utility_to_s1'] for r in rows),
                peak_rss_kib=max(r['peak_rss_kib'] for r in rows)))
    results['run-costs'] = run_costs
    results['individual-checkpoints'] = [row for rows in observed.values() for row in rows]
    return results


def analyze(args):
    study, _, configs, _ = registered()
    final = read(HERE/'final-plan.json')
    names = read(ABLATION/'maze-names.json')
    refs = test_references(configs)
    easy_rows = csv_rows(verified(args.ablation/'registered-analysis/labels.csv', EASY_LABELS))
    easy = {row['task']: row['too_easy']=='True' for row in easy_rows}
    if easy.keys() != configs.keys() or any(row['too_easy'] not in ('True','False') for row in easy_rows):
        raise ValueError('archived ACS2 ease labels differ')
    new, accounting, overlap = measured_sources(args.pilot, args.grid, args.grid_sha, final,
                                                 args.recovery, args.recovery_sha)
    old = old_sources(args.ablation,args.measures,configs,study['seeds'],final)
    observations, validation, timings = raw_inputs(new | old,configs,refs,names)
    results = statistics(study,configs,names,refs,easy,observations,timings)
    carried = carry_share(study,configs,validation)
    result = dict(measured_runs=len(new), reused_pilot=105, reference_runs=len(old),
                  pilot_prefix_checkpoints=overlap, contrasts=len(results['comparisons']),
                  grid_manifest_sha256=args.grid_sha, recovery_manifest_sha256=args.recovery_sha,
                  binary_sha256=BINARY_SHA256,
                  final_plan_sha256=checksum(HERE/'final-plan.json'), accounting=accounting,
                  carried_from_validation=carried, too_easy_source=EASY_SOURCE)
    args.out.mkdir(exist_ok=False,parents=True)
    for name, rows in results.items():
        write_csv(args.out/f'{name}.csv',rows)
    (args.out/'registered-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for arg in ('pilot','grid','ablation','measures','out'):
        parser.add_argument(f'--{arg}',type=Path,required=True)
    parser.add_argument('--grid-sha',required=True)
    parser.add_argument('--recovery',type=Path)
    parser.add_argument('--recovery-sha')
    analyze(parser.parse_args())
