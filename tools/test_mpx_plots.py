from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
from plot_mpx import group_by_seed, plot_reach, plot_signal


def point(**changes):
    return {"source": "one.out", "block": "1", "size": "70", "seed": "42", "repeat": "0",
            "variant": "pyalcs", "encoding": "flip", "epsilon": "0.8", "u_max": "8",
            "agent": "acs2", "do_ga": "true", "trials": 500, "knowledge": 0.5,
            "addr_random": 1.0, "addr_spec": 2.0, "addr_full": 0.1, **changes}


class PlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_mixed_agents_and_replay_parameters_are_rejected(self):
        for change in ({"agent": "acs2er"}, {"do_ga": "false"},
                       {"agent": "acs2er", "er_samples_number": "3"}):
            with self.assertRaises(SystemExit):
                group_by_seed([point(), point(**change)], 70, "pyalcs")
        base = point(agent="acs2er", er_buffer_size="10000", er_min_samples="1000", er_samples_number="1")
        for key in ("er_buffer_size", "er_min_samples", "er_samples_number"):
            with self.assertRaises(SystemExit):
                group_by_seed([base, {**base, key: "10"}], 70, "pyalcs")
        with self.assertRaises(SystemExit):
            group_by_seed([point(), point(seed="43", agent="acs2er")], 70, "pyalcs")

    def test_independent_runs_require_source_or_block_selection(self):
        rows = [point(), point(source="two.out")]
        with self.assertRaises(SystemExit):
            group_by_seed(rows, 70, "pyalcs")
        self.assertEqual(group_by_seed(rows, 70, "pyalcs", sources=["one.out"])[42], rows[:1])
        rows = [point(), point(block="2")]
        self.assertEqual(group_by_seed(rows, 70, "pyalcs", arm={"block": "2"})[42], rows[1:])

    def test_success_marker_cannot_come_from_another_run(self):
        with patch("plot_mpx.save") as save:
            plot_reach([point()], [point(source="other.out", verdict="SUCCESS", trials=1000)],
                       70, "pyalcs", Path("unused"), ["png"])
            axes = save.call_args.args[0].axes[0]
            markers = [line for line in axes.lines if line.get_marker() == "o"]
            self.assertEqual(len(markers), 1)
            self.assertEqual(markers[0].get_markerfacecolor(), "white")
            self.assertEqual(list(markers[0].get_xdata()), [500])

    def test_signal_obeys_variant_seed_arm_and_source(self):
        rows = [point(), point(agent="acs2er", source="er.out"), point(seed="43"),
                point(variant="butz"), point(source="duplicate.out")]
        with patch("plot_mpx.save") as save:
            plot_signal(rows, {70}, Path("unused"), ["png"], "pyalcs", 42,
                        {"agent": "acs2"}, ["one.out"])
            axes = save.call_args.args[0].axes[0]
            self.assertEqual(list(axes.lines[0].get_xdata()), [500])
            self.assertEqual(list(axes.lines[0].get_ydata()), [2.0])


if __name__ == "__main__":
    unittest.main()
