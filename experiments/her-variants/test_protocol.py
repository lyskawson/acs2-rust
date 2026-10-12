import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'grid'))
from collect import validate_row
from row_schema import evaluation_blocks
from submit import run_batches
from matched_control import admit, configurations as control_configurations, reservation_seconds
from protocol import AGENTS, MATCHED_CONTROL, PROPORTIONS, VALUES, configurations, fit_forecast, forecast, pilot, select_proportion, study
from thresholds import THRESHOLDS


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
        thresholds = copy.deepcopy(THRESHOLDS)
        plan = study(thresholds,costs(),1000)
        self.assertEqual(plan['final_grid']['runs'], 600)
        self.assertEqual(plan['seeds'], list(range(42,62)))
        self.assertEqual(plan['tuning']['evaluation_blocks'], ['validation'])
        self.assertEqual(plan['analysis']['thresholds'], thresholds)
        self.assertEqual(len(plan['analysis']['comparisons']), 10)
        with self.assertRaises(ValueError): study({c['id']:0.2 for c in configurations()},costs(),1000)

    def test_budget_counts_pilots_tuning_grid_and_allocation_overhead(self):
        result = forecast(costs(),1000)
        self.assertEqual(result['grid_reservation_cpu_seconds'], 600 * 2 * 60)
        self.assertEqual(result['tuning_reservation_cpu_seconds'], 180 * 2 * 60)
        self.assertAlmostEqual(result['grid_expected_cpu_hours'], 600 * 43 / 3600)
        self.assertAlmostEqual(result['hard_bound_with_pilots_cpu_hours'], (1000 + 780 * 120) / 3600)
        self.assertGreaterEqual(result['memory_mib'], 512)

    def test_forecast_reduces_tuning_first_and_refuses_an_unaffordable_fixed_grid(self):
        choices, result = fit_forecast(costs(100),64800)
        self.assertEqual(choices, PROPORTIONS[:2])
        self.assertLessEqual(result['hard_bound_with_pilots_cpu_hours'],60)
        with self.assertRaises(ValueError): fit_forecast(costs(200),1000)

    def test_batch_size_keeps_twenty_to_forty_jobs_without_host_pinning(self):
        plan = dict(stage='grid_pending_approval',configurations=configurations(),agents=AGENTS,seeds=list(range(42,62)),batch_size=20)
        batches = run_batches(plan)
        self.assertEqual(len(batches),30)
        self.assertTrue(all(len(batch)==20 for batch in batches))
        self.assertEqual(sum(map(len,batches)),600)

    def test_tuning_reserves_the_larger_agent_limit_for_every_element_in_a_maze_group(self):
        cells=costs()
        for key in cells:
            if key[1]=='her_values': cells[key]['body_wall_seconds']=100
        result=forecast(cells,1000)
        self.assertEqual(result['tuning_reservation_cpu_seconds'],6*30*3*60)
        configs=[]
        for config in configurations():
            for p in PROPORTIONS:
                configs.append(dict(config,batch_group=config['id'],proportion=p))
        batches=run_batches(dict(stage='tuning_pending_approval',configurations=configs,agents=VALUES,seeds=list(range(42,47)),batch_by='group'))
        self.assertEqual([len(batch) for batch in batches],[30]*6)

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

    def test_u1_observed_composition_is_required_when_registered_and_has_no_objective_work(self):
        row,config=row_and_configuration('acs2er_u1')
        config['u1_draw_observer']=True
        with self.assertRaises(ValueError): validate_row(row,config,'acs2er_u1',42,'a'*40,1000)

        proxy,_=row_and_configuration('er_episodic')
        diagnostics=proxy['replay_diagnostics']
        updates=row['replay_updates']
        diagnostics.update(draws=updates,original=updates,realized_relabeled_share=0,done=10,
            objective_evaluations=0,reward_evaluations=0,reach_evaluations=0,
            strategies=dict(original=updates,final=0,future=0,episode=0,uniform_real=0),
            draw_observer='core_u1_outcomes_no_extra_random_or_objective_calls')
        diagnostics['routes']['original'].update(draws=updates,done=10,reward_sum=10000,mean_reward=10000/updates)
        for purpose in diagnostics['objective_evaluations_by_purpose'].values():
            purpose.update(objective_evaluations=0,reward_evaluations=0,reach_evaluations=0)
        row['replay_diagnostics']=diagnostics
        validate_row(row,config,'acs2er_u1',42,'a'*40,1000)
        diagnostics['routes']['original']['done']=9
        with self.assertRaises(ValueError): validate_row(row,config,'acs2er_u1',42,'a'*40,1000)

    def test_matched_control_adds_only_mazes_with_a_different_selected_values_proportion(self):
        plan=study(copy.deepcopy(THRESHOLDS),costs(),1000)
        selections={config['id']:{agent:2/3 for agent in VALUES} for config in plan['configurations']}
        self.assertEqual(control_configurations(plan,selections),[])
        for config in plan['configurations'][:2]: selections[config['id']]['her_values']=1/3
        for config in plan['configurations'][2:]: selections[config['id']]['her_values_shortened']=1/3
        controls=control_configurations(plan,selections)
        self.assertEqual([config['id'] for config in controls],[config['id'] for config in plan['configurations'][:2]])
        self.assertEqual(reservation_seconds(controls),40*2*60)
        for config,original in zip(controls,plan['configurations']):
            self.assertEqual(config['row_arm'],MATCHED_CONTROL)
            self.assertEqual(config['agent_settings'],{'her_values':{'relabeled_proportion':2/3}})
            self.assertEqual({key:value for key,value in config.items() if key not in ('row_arm','agent_settings')},original)
        row,config=row_and_configuration('her_values')
        config['row_arm']=MATCHED_CONTROL
        row['arm']=MATCHED_CONTROL
        validate_row(row,config,'her_values',42,'a'*40,1000)
        row['arm']='her_values_tuned'
        with self.assertRaises(ValueError): validate_row(row,config,'her_values',42,'a'*40,1000)
        comparisons=plan['analysis']['comparisons']
        self.assertEqual([pair for pair in comparisons if MATCHED_CONTROL in pair],[[MATCHED_CONTROL,'her_full']])
        self.assertIn(['her_values','her_full'],comparisons)
        with self.assertRaises(ValueError): control_configurations(plan,{})
        selections[plan['configurations'][0]['id']]['her_values']=0.6
        with self.assertRaises(ValueError): control_configurations(plan,selections)

    def test_control_admission_counts_all_batches_and_waits_for_released_reservations(self):
        registered=json.loads((Path(__file__).with_name('plan.json')).read_text())
        selections={config['id']:{agent:1/3 for agent in VALUES} for config in registered['configurations']}
        controls=control_configurations(registered,selections)
        self.assertEqual(reservation_seconds(controls),34800)
        self.assertEqual([reservation_seconds([config]) for config in controls],[4800,7200,8400,8400,3600,2400])
        forecasted=registered['forecast']
        states=[dict(spent_seconds=0,reserved_seconds=forecasted['grid_reservation_cpu_seconds']),
                dict(spent_seconds=0,reserved_seconds=forecasted['tuning_reservation_cpu_seconds'])]
        with self.assertRaises(ValueError): admit(5445,states,controls)
        states[0]=dict(spent_seconds=82000,reserved_seconds=0)
        states[1]=dict(spent_seconds=18500,reserved_seconds=0)
        admitted=admit(5445,states,controls)
        self.assertEqual(admitted['additional_runs'],120)
        self.assertEqual(admitted['bound_seconds'],5445+82000+18500+34800)
        with self.assertRaises(ValueError): admit(5445,states+[dict(spent_seconds=80000,reserved_seconds=0)],controls)
        with self.assertRaises(ValueError): admit(5445,[dict(spent_seconds=-1,reserved_seconds=0)],controls)



if __name__ == '__main__': unittest.main()
