import json
import subprocess

from pair_sources import (ARMS, E1, GRID_MANIFEST, FINAL_PLAN, GROUPS, HERE, LAMBDAS, MEAN, PAIR_SEEDS, PER_PAIR,
                          PROPORTIONS, RANDOM, RECOVERY_MANIFEST, REPO, RUNS, S1, SEEDS, STUDY, check_sources,
                          registered_configs, select_sources, sha256, truth_model)

BINS = (('1-2', 1, 2), ('3-4', 3, 4), ('5-6', 5, 6), ('7+', 7, None))
MEASURES = ('F', 'fallback_share', 'value_error', 'absolute_value_error', 'overestimation', 'success')
COUNTERS = ('trajectory_utility.diagnostics.distinct_assigned_goals',
            'trajectory_utility.diagnostics.assigned_goal_reached_share',
            'trajectory_utility.diagnostics.updates_per_goal.minimum',
            'trajectory_utility.diagnostics.updates_per_goal.median',
            'trajectory_utility.diagnostics.updates_per_goal.maximum',
            'trajectory_utility.diagnostics.realized_copy_shares.hindsight',
            'trajectory_utility.diagnostics.realized_copy_shares.trajectory_utility',
            'trajectory_utility.diagnostics.realized_copy_shares.total',
            'trajectory_utility.diagnostics.hindsight_fallbacks',
            'trajectory_utility.diagnostics.tu_fallbacks',
            'replay_diagnostics.realized_relabeled_share',
            'replay_diagnostics.fallbacks')


def arm(agent, share):
    return f'{agent}_{dict(LAMBDAS)[share]}'


def families():
    result = [dict(family='her_copies', left=S1, right=E1, tag='S+1 - E+1')]
    for share, _ in LAMBDAS:
        result.append(dict(family='tu_copies', share=share, left=arm(RANDOM, share), right=S1, tag=f'R {share} - S+1'))
        result.append(dict(family='tu_copies', share=share, left=arm(MEAN, share), right=S1, tag=f'M {share} - S+1'))
    for share, _ in LAMBDAS:
        result.append(dict(family='goal_choice', share=share, left=arm(RANDOM, share), right=arm(MEAN, share),
                           tag=f'R {share} - M {share}'))
    return result


def cells(contrast, measure, sign, bins, checkpoints):
    return [dict(group='no_aliasing', contrast=contrast, measure=measure, predicted_sign=sign, bin=item,
                 checkpoint=point) for item in bins for point in checkpoints]


def predictions():
    far, near, late, early = ('5-6', '7+'), ('1-2', '3-4'), (30000, 100000), (1000, 3000, 10000)
    return [
        dict(id='P1', text='R - S+1 in F is positive on pairs of 5 or more steps at 30k and 100k, at lambda = 1/2 and 3/4.',
             cells=[cell for share in ('1/2', '3/4') for cell in cells(f'R {share} - S+1', 'F', 1, far, late)]),
        dict(id='P2', text='R - S+1 in the signed value error is positive on pairs of 5 or more steps at 30k and 100k, '
                           'at lambda = 1/2 and 3/4. The absolute error is reported beside it without a prediction.',
             cells=[cell for share in ('1/2', '3/4') for cell in cells(f'R {share} - S+1', 'value_error', 1, far, late)]),
        dict(id='P3', text='R - S+1 in F is negative on pairs of 1-4 steps at 1k, 3k and 10k at lambda = 1.',
             cells=cells('R 1 - S+1', 'F', -1, near, early)),
        dict(id='P4', text='S+1 - E+1 in F is positive on pairs of 1-4 steps at 1k, 3k and 10k.',
             cells=cells('S+1 - E+1', 'F', 1, near, early)),
        dict(id='P5', text='R - M in F is positive on pairs of 5 or more steps at 30k and 100k, at lambda = 3/4.',
             cells=cells('R 3/4 - M 3/4', 'F', 1, far, late)),
    ]


def definitions():
    return dict(
        post_hoc='Everything here is post hoc relative to the runs. It describes associations in stored rows, not causes.',
        pairs='The registered test block of every maze, every pair at the registered cap L, recorded weights checked equal '
              'within the maze; shortest distance d recorded per pair and checked against the maze geometry.',
        bins=[dict(name=name, low=low, high=high) for name, low, high in BINS],
        exact_distances='Every exact d is also tabulated, in JSON and CSV.',
        pooling='Within a maze each pair has equal weight. A pooled group view gives every pair weight 1 across the mazes of '
                'the group that have the checkpoint, so larger mazes contribute more pairs. Pooling happens within one '
                'arm, seed and checkpoint; seeds are never mixed.',
        truth='Q*(s, a, g) and V*(s, g) from acs2-maze-truth <maze> <L>, attached by per-pair-diagnostics analysis.attach '
              '(unchanged code, imported), which stops when the recorded distance and the recomputed V* disagree.',
        measures=dict(
            F='first action in the optimal-action set of the capped truth, over non-fallback first decisions only '
              '(per-pair-diagnostics statistics_for F); undefined when a cell has no non-fallback decision',
            fallback_share='share of pairs whose first decision is a random fallback, over all pairs',
            value_error='estimated_first_action_value - Q*(s, chosen first action, g), over all pairs including '
                        'fallbacks (per-pair-diagnostics values.all.estimate_minus_q)',
            absolute_value_error='|estimated_first_action_value - Q*(s, chosen first action, g)|, same pairs',
            overestimation='estimated_first_action_value > V*(s, g), over non-fallback first decisions '
                           '(per-pair-diagnostics decision_metrics estimate_above_v)',
            success='recorded success at cap L'),
        comparisons=[dict(item) for item in families()],
        interval=dict(statistic='per seed: pooled measure of the left arm minus pooled measure of the right arm in the '
                                'same group, distance bin and checkpoint; mean over seeds 42-46',
                      degrees_of_freedom=4, critical_value=2.7764451051977987,
                      implementation='per-pair-diagnostics analysis.paired, unchanged',
                      missing='a seed with an undefined measure makes the comparison undefined, reported with its count'),
        counters=dict(fields=list(COUNTERS), seeds=list(SEEDS),
                      rule='Values as recorded in each row, per arm, maze and checkpoint, summarized over seeds. A field '
                           'absent from an arm is reported as not recorded, never derived.'),
        excluded='Trial update is not carried (decision 60).')


def main():
    target = HERE / 'plan.json'
    if target.exists():
        raise SystemExit('plan.json exists; the registration is immutable')
    configs = registered_configs()
    sources = select_sources(configs)
    universes = check_sources(sources, configs)
    truth = {}
    for task, config in configs.items():
        _, digest = truth_model(task, config['cap'])
        truth[task] = dict(cap=config['cap'], sha256=digest)
    commit = subprocess.check_output(['/usr/bin/git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    repository = {str(path.relative_to(REPO)): sha256(path) for path in (
        PER_PAIR / 'analysis.py', PER_PAIR / 'decision_metrics.py', PER_PAIR / 'post_hoc.py', PER_PAIR / 'protocol.py',
        REPO / 'experiments/grid/pair_schema.py', REPO / 'experiments/per-goal-reward-ablation/plan.json',
        REPO / 'experiments/per-goal-reward-ablation/maze-names.json', PROPORTIONS / 'grid_results.py',
        PROPORTIONS / 'final-plan.json', REPO / 'acs2-measure/src/maze_truth.rs',
        REPO / 'acs2-measure/src/bin/acs2-maze-truth.rs')}
    plan = dict(
        experiment=STUDY, stage='registration_before_any_computed_number', post_hoc=True,
        question='Whether random TU goals improve the first decision and its value estimate on far pairs earlier than '
                 'S+1 does, what HER copies add over the same agent without copies, and how Mean value differs from '
                 'Random goal, on stored per-pair test records without new runs.',
        registration_commit_parent=commit,
        arms=dict(E1=E1, S1=S1, M={share: arm(MEAN, share) for share, _ in LAMBDAS},
                  R={share: arm(RANDOM, share) for share, _ in LAMBDAS}),
        groups=list(GROUPS), mazes={task: dict(group=config['group'], cap=config['cap'], checkpoints=config['targets'],
                                               test_pairs=config['registration']['blocks']['test']['evaluation_pairs'])
                                    for task, config in configs.items()},
        seeds=dict(per_pair=list(PAIR_SEEDS), counters=list(SEEDS)),
        definitions=definitions(), predictions=predictions(),
        criterion='A prediction cell is supported when the pooled mean difference has the predicted sign and its t(4) '
                  'interval excludes zero on that side, contradicted when the interval excludes zero on the other side, '
                  'and inconclusive otherwise, including an undefined cell. Each prediction is judged in every cell it '
                  'names: 5 or more steps means bins 5-6 and 7+ separately, 1-4 steps means bins 1-2 and 3-4 separately. '
                  'The report gives the count of each verdict per prediction.',
        source_selection=dict(
            fixed_share='trajectory-utility-proportions grid_results.measured_sources: pilot, first submission and '
                        'recovery from the sealed manifests',
            grid_manifest_sha256=GRID_MANIFEST, recovery_manifest_sha256=RECOVERY_MANIFEST, final_plan_sha256=FINAL_PLAN,
            references='E+1 and S+1 from the per-goal-reward-ablation local-full-index (six pyalcs mazes resolve to the '
                       'per-goal-reward archive, 22 ALCS mazes to per-goal-reward-ablation); M and R at lambda = 1/2 from '
                       'the trajectory-utility-measures registered full-index; both checked by the checksums the '
                       'fixed-share analysis uses',
            archive_root='~/Desktop/tu-runs', stop='Any run without complete per-pair test records on seeds 42-46 '
                                                   'stops the analysis; nothing is substituted.'),
        repository_sha256=repository, truth_models=truth, test_pair_universe_sha256=universes,
        inputs_fields=['task', 'arm', 'seed', 'archive', 'path', 'sha256', 'kind'],
        inputs=[[source[name] for name in ('task', 'arm', 'seed', 'archive', 'path', 'sha256', 'kind')]
                for source in sources],
        outputs=str(RUNS / STUDY))
    inputs = plan.pop('inputs')
    head = json.dumps(plan, indent=1)
    rows = ',\n'.join('  ' + json.dumps(row) for row in inputs)
    target.write_text(head[:-2] + ',\n "inputs": [\n' + rows + '\n ]\n}\n')
    if json.loads(target.read_text())['inputs'] != inputs:
        raise ValueError('registered inputs do not round-trip')
    print(json.dumps(dict(inputs=len(sources), mazes=len(configs), plan_sha256=sha256(target))))


if __name__ == '__main__':
    main()
