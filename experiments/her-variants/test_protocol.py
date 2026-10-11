import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'grid'))
from collect import validate_row
from row_schema import evaluation_blocks
from submit import run_batches
from protocol import AGENTS, PROPORTIONS, VALUES, configurations, fit_forecast, forecast, pilot, select_proportion, study


def costs(seconds=40):
    return {(config['id'], agent): dict(body_wall_seconds=seconds, allocation_cpu_seconds=seconds + 3,
            process_cpu_seconds=seconds, peak_rss_kib=40000) for config in configurations() for agent in AGENTS}


def row_and_configuration(agent):
    root = Path(__file__).resolve().parents[2]
    row = json.loads((root / 'fixtures/pair_baseline_regression.json').read_text())['configurations'][0]['rows'][0]
    row.update(commit='a' * 40, source_state='clean', host='lem', cpu_model='AMD EPYC 9554 64-Core Processor', agent=agent)
    config = configurations()[0]
    steps = row['actual_steps']
    updates = steps if agent == 'acs2er_u1' else 3 * steps
    row.update(online_updates=0, replay_updates=updates, learning_update_counts=dict(full=updates, values_only=0, skipped_warmup_steps=0))
    parameters = dict(replay=True, replay_capacity=10000, replay_warmup=1, replay_updates_per_step=1 if agent == 'acs2er_u1' else 3)
    if agent == 'acs2er_u1':
        row['agent_parameters'] = parameters
        row.update(trajectory_logical_bytes=0, replay_diagnostics=None)
        return row, config
    parameters.update(schedule='after_complete_episode', strategy='future', candidate_filter=False,
        draws='uniform_transitions_with_replacement', truncation='Bootstrap', sampler_truncation='Bootstrap', sampler_stream=7,
        learning_clock='episode_end_environment_steps', warmup_rule='retained_steps_after_episode_insertion_at_least_warmup_no_backpay',
        admissibility='CounterfactualEpisode' if agent == 'her_values_shortened' else 'EveryTransition',
        copy_update='Full' if agent in ('er_episodic','her_full') else 'ValuesOnly', relabeled_proportion=0 if agent == 'er_episodic' else 2 / 3)
    row['agent_parameters'] = parameters
    relabeled = 0 if agent == 'er_episodic' else updates // 2
    original = updates - relabeled
    if agent in VALUES:
        row['learning_update_counts'].update(full=original, values_only=relabeled)
    route = lambda count: dict(draws=count, already_reached=0, after_counterfactual_end=0, done=0, outside_candidates=0, reward_sum=0, mean_reward=0 if count else None)
    cost = lambda rewards, reaches: dict(reward_evaluations=rewards, reach_evaluations=reaches, objective_evaluations=rewards + reaches)
    row.update(replay_samples=0, replay_logical_bytes=0, trajectory_logical_bytes=50000,
        replay_diagnostics=dict(draws=updates, failed_draws=0, original=original, relabeled=relabeled, fallbacks=0,
        already_reached=0, after_counterfactual_end=0, done=0, outside_candidates=0, realized_relabeled_share=relabeled / updates,
        stored_steps=steps, stored_episodes=100, routes=dict(original=route(original), relabeled=route(relabeled)),
        reward_evaluations=updates, reach_evaluations=updates, objective_evaluations=2 * updates,
        objective_evaluations_by_purpose=dict(scoring=cost(updates,updates), selection=cost(0,0), provenance=cost(0,0))))
    return row, config


class ProtocolTests(unittest.TestCase):
    def test_pilot_has_all_five_agents_and_registered_budgets(self):
        plan = pilot()
        self.assertEqual(plan['agents'], AGENTS)
        self.assertEqual(plan['seeds'], [42])
        self.assertEqual(sum(len(c['targets']) * 5 for c in plan['configurations']), 245)
        self.assertEqual([c['targets'][-1] for c in plan['configurations']], [2000000] * 4 + [4000000,2000000])
        self.assertEqual(len(run_batches(plan)[0]), 30)

    def test_selection_uses_validation_scores_and_registered_tie(self):
        self.assertEqual(select_proportion(dict(zip(PROPORTIONS,[0.1,0.3,0.2]))), 2 / 3)
        self.assertEqual(select_proportion(dict(zip(PROPORTIONS,[0.3,0.3 + 5e-13,0.2]))), 1 / 3)
        with self.assertRaises(ValueError): select_proportion({1/3:0.1})
        with self.assertRaises(ValueError): select_proportion(dict(zip(PROPORTIONS,[float('nan'),0.3,0.2])))

    def test_final_grid_cannot_shrink_agents_mazes_or_seeds(self):
        thresholds = {c['id']:0.2 for c in configurations()}
        plan = study(thresholds,costs(),1000)
        self.assertEqual(plan['final_grid']['runs'], 600)
        self.assertEqual(plan['seeds'], list(range(42,62)))
        self.assertEqual(plan['tuning']['evaluation_blocks'], ['validation'])
        self.assertEqual(plan['analysis']['thresholds'], thresholds)
        self.assertEqual(len(plan['analysis']['comparisons']), 9)

    def test_budget_counts_pilots_tuning_grid_and_allocation_overhead(self):
        result = forecast(costs(),1000)
        self.assertEqual(result['grid_reservation_cpu_seconds'], 600 * 3 * 60)
        self.assertEqual(result['tuning_reservation_cpu_seconds'], 180 * 3 * 60)
        self.assertAlmostEqual(result['grid_expected_cpu_hours'], 600 * 43 / 3600)
        self.assertAlmostEqual(result['hard_bound_with_pilots_cpu_hours'], (1000 + 780 * 180) / 3600)
        self.assertGreaterEqual(result['memory_mib'], 512)

    def test_forecast_reduces_tuning_first_and_refuses_an_unaffordable_fixed_grid(self):
        choices, result = fit_forecast(costs(60),36000)
        self.assertEqual(choices, PROPORTIONS[:2])
        self.assertLessEqual(result['hard_bound_with_pilots_cpu_hours'],60)
        with self.assertRaises(ValueError): fit_forecast(costs(100),1000)

    def test_batch_size_keeps_twenty_to_forty_jobs_without_host_pinning(self):
        plan = dict(stage='grid_pending_approval',configurations=configurations(),agents=AGENTS,seeds=list(range(42,62)),batch_size=20)
        batches = run_batches(plan)
        self.assertEqual(len(batches),30)
        self.assertTrue(all(len(batch)==20 for batch in batches))
        self.assertEqual(sum(map(len,batches)),600)

    def test_every_new_arm_passes_the_shared_row_validator(self):
        for agent in AGENTS:
            row,config = row_and_configuration(agent)
            validate_row(row,config,agent,42,'a'*40,1000)

    def test_validator_catches_wrong_copy_update_counts_goal_end_and_cost_fields(self):
        row,config = row_and_configuration('her_values')
        mutations = [('learning_update_counts','values_only',0), ('agent_parameters','sampler_truncation','Pyalcs'),
                     ('agent_parameters','copy_update','Full'), ('replay_diagnostics','realized_relabeled_share',2/3),
                     ('replay_diagnostics','draws',row['replay_updates'] - 1), ('replay_diagnostics','objective_evaluations',0)]
        for container,key,value in mutations:
            modified = copy.deepcopy(row)
            modified[container][key]=value
            with self.assertRaises(ValueError): validate_row(modified,config,'her_values',42,'a'*40,1000)

    def test_validation_only_rows_require_an_explicit_validation_manifest(self):
        row,config = row_and_configuration('her_values')
        row['evaluation_blocks']={'validation':row['evaluation_blocks']['validation']}
        with self.assertRaises(ValueError): validate_row(row,config,'her_values',42,'a'*40,1000)
        config['evaluation_blocks']=['validation']
        validate_row(row,config,'her_values',42,'a'*40,1000)
        with self.assertRaises(ValueError): evaluation_blocks(row)

    def test_tuning_proportions_are_validated_against_the_assignment(self):
        row,config = row_and_configuration('her_values_shortened')
        config['agent_settings']={'her_values_shortened':{'relabeled_proportion':0.9}}
        with self.assertRaises(ValueError): validate_row(row,config,'her_values_shortened',42,'a'*40,1000)
        row['agent_parameters']['relabeled_proportion']=0.9
        validate_row(row,config,'her_values_shortened',42,'a'*40,1000)


if __name__ == '__main__': unittest.main()
