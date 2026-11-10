import csv
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import success_curves as curves
from protocol import ARMS,ROOT


class SuccessCurveTests(unittest.TestCase):
    def fixture(self,root):
        configs=[dict(task='synthetic',group='no_aliasing',aliased_share=0,targets=[1000,10000,2000000])]
        names=dict(synthetic='Synthetic maze')
        rows=[dict(task='synthetic',maze='Synthetic maze',group='no_aliasing',arm=arm,nominal_step=step,n=20,
            mean=.10+index*.08+point*.12) for index,arm in enumerate(ARMS) for point,step in enumerate(configs[0]['targets'])]
        path=root/'curves.csv';self.write(path,rows)
        return path,rows,configs,names

    def write(self,path,rows):
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    def test_curve_input_rejects_missing_arms_swapped_groups_names_counts_and_duplicate_points(self):
        with tempfile.TemporaryDirectory() as directory:
            path,rows,configs,names=self.fixture(Path(directory))
            self.assertEqual(len(curves.read_curves(path,configs,names)),18)
            for field,value in [('group','aliasing'),('maze','Other maze'),('n',19),('mean',1.1),('nominal_step',3000)]:
                changed=[dict(r) for r in rows];changed[0][field]=value;self.write(path,changed)
                with self.assertRaises(ValueError):curves.read_curves(path,configs,names)
            for changed in [rows[:-1],rows+[rows[0]]]:
                self.write(path,changed)
                with self.assertRaises(ValueError):curves.read_curves(path,configs,names)

    def test_plot_driver_writes_every_arm_and_both_individual_and_grouped_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,rows,configs,names=self.fixture(root)
            config=root/'config.json';config.write_text(json.dumps(dict(configs=configs,names=names)))
            output=root/'figures'
            code="import sys,json;sys.path.insert(0,sys.argv[1]);import success_curves as s;v=json.load(open(sys.argv[2]));print(json.dumps(s.plot(sys.argv[3],sys.argv[4],v['configs'],v['names'])))"
            result=subprocess.run(['uv','run','--project',str(ROOT/'tools'),'python','-B','-c',code,
                str(Path(curves.__file__).parent),str(config),str(path),str(output)],capture_output=True,text=True,check=True)
            metadata=json.loads(result.stdout)
            self.assertEqual(metadata['arms'],list(ARMS));self.assertEqual(metadata['group_pages'],1)
            self.assertEqual(metadata['groups'],dict(no_aliasing=['synthetic']))
            for name in ['no_aliasing/synthetic.png','no_aliasing/synthetic.pdf','no_aliasing-test-success.pdf','no_aliasing-page-1.png']:
                self.assertGreater((output/name).stat().st_size,1000)


if __name__=='__main__':unittest.main()
