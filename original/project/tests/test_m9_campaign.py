from pathlib import Path
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

class CampaignTests(unittest.TestCase):
    def module(self):
        try:return importlib.import_module("tools.m9_campaign")
        except ModuleNotFoundError:self.fail("M9 campaign supervisor is not implemented")
    def fixture(self):
        m=self.module()
        config=json.loads((Path(__file__).resolve().parents[1]/"configs/m9_research_attempt1.json").read_text(encoding="utf-8"))
        return m,dict(config=config,registration_sha256="a"*64,test_seeds=list(range(3400000,3400100)))
    def test_paired_runs_equal_training_draw_budget_and_fresh_seeds(self):
        m,reg=self.fixture()
        a=m.task_spec(reg,"explore","control",0)
        b=m.task_spec(reg,"explore","infinite_freeze",0)
        c=m.task_spec(reg,"explore","control",1)
        self.assertEqual(a["train_seed_start"],b["train_seed_start"])
        self.assertEqual(a["rng_seed"],b["rng_seed"])
        self.assertEqual(a["evaluation_seeds"],b["evaluation_seeds"])
        self.assertEqual(a["updates"]*a["envs"]*a["rollout_steps"],16384)
        self.assertNotEqual(a["train_seed_start"],c["train_seed_start"])
        self.assertTrue(set(a["evaluation_seeds"]).isdisjoint(reg["test_seeds"]))
        m.validate_task(reg,a)
        a["evaluation_seeds"]=reg["test_seeds"][:8]
        with self.assertRaises(ValueError):m.validate_task(reg,a)
    def test_config_rejects_unregistered_world_or_increased_budget(self):
        m,reg=self.fixture()
        task=m.task_spec(reg,"explore","control",0)
        task["updates"]+=1
        with self.assertRaises(ValueError):m.validate_task(reg,task)
        task=m.task_spec(reg,"explore","control",0);task["map_mode"]="infinite_without_needs"
        with self.assertRaises(ValueError):m.validate_task(reg,task)
    def test_calibration_uses_full_training_batch_and_separate_worlds(self):
        m,reg=self.fixture()
        task=m.task_spec(reg,"calibration","control",0)
        self.assertEqual(task["updates"]*task["envs"]*task["rollout_steps"],4096)
        self.assertEqual(task["horizon"],64)
        formal=m.task_spec(reg,"explore","control",0)
        self.assertTrue(set(task["evaluation_seeds"]).isdisjoint(formal["evaluation_seeds"]))
    @unittest.skipUnless(os.name=="posix","Owned Linux process groups")
    def test_failed_child_descendant_is_reaped(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            script="import subprocess,sys,pathlib; child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']); pathlib.Path(sys.argv[1]).write_text(str(child.pid));sys.exit(3)"
            code=m.run_owned([sys.executable,"-c",script,str(p/"pid")],p,p/"stage",dict(os.environ),time.monotonic()+20,lambda:False)
            self.assertEqual(code,3)
            pid=int((p/"pid").read_text())
            stat=Path(f"/proc/{pid}/stat")
            self.assertTrue(not stat.exists() or stat.read_text().split(") ")[1].startswith("Z "))
    @unittest.skipUnless(os.name=="posix","Owned Linux process groups")
    def test_deadline_stops_only_owned_worker(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            with self.assertRaises(TimeoutError):
                m.run_owned([sys.executable,"-c","import time;time.sleep(120)"],p,p/"deadline",dict(os.environ),time.monotonic()+.2,lambda:False)
            self.assertTrue((p/"deadline.json").exists())

if __name__=="__main__":unittest.main()

