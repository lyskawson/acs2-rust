import copy
from pathlib import Path
import unittest

from execution import approved, choose, control_manifest, grid_manifest, label, references, tuning_manifest
from protocol import AGENTS, VALUES
from submit import run_batches


class ExecutionTests(unittest.TestCase):
    def plan(self):
        return approved(Path(__file__).with_name('plan.json'))

    def selection(self,plan):
        return dict(selections={config['id']:{agent:dict(proportion=2/3) for agent in VALUES}
                                for config in plan['configurations']})

    def test_tuning_derives_exactly_the_registered_validation_scope_and_batch_limits(self):
        plan=self.plan()
        manifest=tuning_manifest(plan)
        self.assertEqual(manifest['agents'],VALUES)
        self.assertEqual(manifest['seeds'],list(range(42,47)))
        self.assertEqual([len(batch) for batch in run_batches(manifest)],[20]*6)
        self.assertEqual(sum(len(config['targets'])*10 for config in manifest['configurations']),980)
        reserved=sum(len(batch)*60*max(config['limits_minutes'][agent] for config,agent,_ in batch) for batch in run_batches(manifest))
        self.assertEqual(reserved,34800)
        for config in manifest['configurations']:
            original=next(item for item in plan['configurations'] if item['id']==config['registered_id'])
            for field in ['registration','cap','budget_steps','targets','knowledge_transition_goal_pairs']:
                self.assertEqual(config[field],original[field])
            self.assertEqual(config['evaluation_blocks'],['validation'])

    def test_grid_and_control_use_twenty_final_seeds_and_no_extra_control_for_selected_two_thirds(self):
        plan=self.plan()
        selected=self.selection(plan)
        grid=grid_manifest(plan,selected)
        self.assertEqual((grid['agents'],grid['seeds']),(AGENTS,list(range(42,62))))
        self.assertEqual(sum(map(len,run_batches(grid))),600)
        identifier=plan['configurations'][0]['id']
        with self.assertRaises(ValueError): control_manifest(plan,selected,identifier)
        selected['selections'][identifier]['her_values']['proportion']=1/3
        extra=control_manifest(plan,selected,identifier)
        self.assertEqual(extra['seeds'],list(range(42,62)))
        self.assertEqual(extra['configurations'][0]['row_arm'],'her_values_p2of3')
        self.assertNotIn('evaluation_blocks',extra['configurations'][0])
        self.assertEqual(extra['configurations'][0]['agent_settings'],{'her_values':{'relabeled_proportion':2/3}})

    def test_selection_and_identity_references_are_mechanical_and_use_matching_proportions(self):
        plan=self.plan()
        entries={}
        pilots={}
        for config in plan['configurations']:
            for agent in AGENTS:
                pilots[config['id'],agent,42]=dict(path='pilot-'+agent,sha256='pilot',targets=config['targets'])
            for proportion in plan['tuning']['proportions']:
                for agent in VALUES:
                    for seed in plan['tuning']['seeds']:
                        entries[config['id']+'_'+label(proportion),agent,seed]=dict(path=f'tuning-{agent}-{proportion}-{seed}',sha256='tuning',targets=config['targets'],
                            rows=[dict(agent_parameters=dict(relabeled_proportion=proportion),seed=seed,nominal_step=point,
                                       evaluation_blocks={'validation':{'success':proportion}}) for point in config['targets']])
        selection=choose(plan,entries)
        self.assertTrue(all(result['proportion']==2/3 for values in selection['selections'].values() for result in values.values()))
        refs=references(grid_manifest(plan,selection),pilots,entries)
        self.assertEqual(len(refs),78)
        identifier=plan['configurations'][0]['id']
        key=identifier+'|her_values|42'
        self.assertEqual([ref['blocks'] for ref in refs[key]],[['training','validation','test'],['validation']])
        self.assertIn('0.6666666666666666',refs[key][1]['path'])
        changed=copy.deepcopy(selection)
        changed['selections'][identifier]['her_values']['proportion']=1/3
        refs=references(grid_manifest(plan,changed),pilots,entries)
        self.assertEqual([ref['blocks'] for ref in refs[key]],[['validation']])
        self.assertIn('0.3333333333333333',refs[key][0]['path'])


if __name__ == '__main__': unittest.main()
