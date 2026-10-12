import copy
from pathlib import Path
import unittest

from analyze import matched_entries, reference_resolver, require_derivation


class ArtifactAnalysisTests(unittest.TestCase):
    def test_matched_two_thirds_reuses_twenty_final_grid_seeds_and_three_blocks(self):
        main={('maze','her_values',seed):dict(source='grid',rows=[dict(evaluation_blocks={name:{} for name in ['training','validation','test']})]) for seed in range(42,62)}
        extra={('maze','her_values',seed):dict(source='control',rows=main['maze','her_values',seed]['rows']) for seed in range(42,62)}
        group=matched_entries('maze',2/3,main,extra,list(range(42,62)))
        self.assertEqual(len(group),20)
        self.assertTrue(all(entry['source']=='grid' for entry in group))
        group=matched_entries('maze',1/3,main,extra,list(range(42,62)))
        self.assertTrue(all(entry['source']=='control' for entry in group))
        with self.assertRaises(ValueError): matched_entries('maze',2/3,main,extra,list(range(42,47)))
        changed=copy.deepcopy(main)
        changed['maze','her_values',42]['rows'][0]['evaluation_blocks']={'validation':{}}
        with self.assertRaises(ValueError): matched_entries('maze',2/3,changed,extra,list(range(42,62)))

    def test_local_verification_resolves_only_owned_reference_roots_without_rewriting_manifests(self):
        resolve=reference_resolver(Path('/local/study'),Path('/local/pilots'))
        self.assertEqual(resolve('/home/alelys2099/tu_runs/5/approved-her-study/tuning/rows.jsonl'),Path('/local/study/tuning/rows.jsonl'))
        self.assertEqual(resolve('/home/alelys2099/tu_runs/5/cost-pilot/rows.jsonl'),Path('/local/pilots/cost-pilot/rows.jsonl'))
        with self.assertRaises(ValueError): resolve('/home/alelys2099/mpx_runs/rows.jsonl')

    def test_analysis_rejects_nonmechanical_execution_manifests(self):
        expected=dict(seeds=list(range(42,62)),agents=['her_values'],configurations=[dict(cap=5,proportion=2/3)])
        require_derivation(dict(expected,metadata='extra'),expected)
        changed=copy.deepcopy(expected)
        changed['configurations'][0]['proportion']=1/3
        with self.assertRaises(ValueError): require_derivation(changed,expected)


if __name__ == '__main__': unittest.main()
