import copy
import unittest

from protocol import AGENTS, PROPORTIONS, configurations
from summarize_check import summarize
from test_protocol import row_and_configuration
from tuning import select


class MeasurementTests(unittest.TestCase):
    def test_validation_tuning_averages_points_inside_seed_and_never_reads_test(self):
        rows=[]
        for p in PROPORTIONS:
            for seed in range(42,47):
                for point in [1000,3000]:
                    rows.append(dict(agent_parameters=dict(relabeled_proportion=p),seed=seed,nominal_step=point,
                                     evaluation_blocks=dict(validation=dict(success=p/2))))
        result=select(rows,PROPORTIONS,[1000,3000])
        self.assertEqual(result['proportion'],0.9)
        self.assertEqual(result['validation_scores'][-1]['mean'],0.45)
        with self.assertRaises(ValueError): select(rows[:-1],PROPORTIONS,[1000,3000])
        rows[0]['evaluation_blocks']['test']=dict(success=1)
        with self.assertRaises(ValueError): select(rows,PROPORTIONS,[1000,3000])

    def test_local_check_has_five_seed_errors_and_does_not_rank(self):
        rows=[]
        for config in configurations():
            for agent in AGENTS:
                for seed in range(42,47):
                    for point in [1000,10000,20000]:
                        row,_=row_and_configuration(agent)
                        row.update(task=config['task'],seed=seed,nominal_step=point)
                        row['evaluation_blocks']={'validation':dict(row['evaluation_blocks']['validation'],success=(seed-42)/10)}
                        row.update(wall_seconds_train=1,source_state='clean')
                        rows.append(row)
        result=summarize(rows)
        self.assertEqual((result['runs'],result['rows']),(150,450))
        self.assertAlmostEqual(result['cells'][0]['success']['mean'],0.2)
        self.assertAlmostEqual(result['cells'][0]['success']['se'],0.07071067811865475)
        self.assertIn('no ranking',result['interpretation'])
        with self.assertRaises(ValueError): summarize(rows[:-1])
        with self.assertRaises(ValueError): summarize(rows+[rows[0]])
        modified=copy.deepcopy(rows)
        modified[0]['evaluation_blocks']['test']={}
        with self.assertRaises(ValueError): summarize(modified)


if __name__ == '__main__': unittest.main()
