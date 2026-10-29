import unittest

from agent_names import ARCHIVED_RUN_NAMES
from replay_schema import registered_axes


UNMEASURED_AGENTS = (
    'acs2', 'acs2er', 'acs2er_one_update', 'acs2her', 'acs2her_nontrivial',
    'acs2her_values', 'acs2her_values_shortened', 'acs2vcp_k2', 'acs2vcp_k4',
    'acs2vcp_k2_uniform', 'acs2vcp_k4_uniform',
)
REFUSAL = 'replay axes are supported only for acs2her_shortened and acs2er_episodic'


class ReplaySchemaTests(unittest.TestCase):
    def test_schedules_refuse_unmeasured_agents(self):
        aliases = [name for name, canonical in ARCHIVED_RUN_NAMES.items()
                   if canonical in UNMEASURED_AGENTS]
        for agent in (*UNMEASURED_AGENTS, *aliases):
            for schedule in ('constant', '3to1at30000', '3to1at100000', '3to1at300000'):
                for settings in [dict(replay_schedule=schedule),
                                 dict(replay_schedule=schedule, replay_order='independent')]:
                    with self.subTest(agent=agent, settings=settings), self.assertRaisesRegex(ValueError, REFUSAL):
                        registered_axes(dict(agent_settings={agent: settings}), agent)

    def test_each_order_refuses_unmeasured_agents(self):
        aliases = [name for name, canonical in ARCHIVED_RUN_NAMES.items()
                   if canonical in UNMEASURED_AGENTS]
        for agent in (*UNMEASURED_AGENTS, *aliases):
            for order in ('independent', 'backward', 'random-pass'):
                for settings in [dict(replay_order=order),
                                 dict(replay_schedule='constant', replay_order=order)]:
                    with self.subTest(agent=agent, settings=settings), self.assertRaisesRegex(ValueError, REFUSAL):
                        registered_axes(dict(agent_settings={agent: settings}), agent)

    def test_measured_agents_keep_all_registered_schedules_and_pass_orders(self):
        for agent in ('acs2her_shortened', 'acs2er_episodic', 'er_episodic'):
            for schedule, order in [('3to1at30000', 'independent'),
                                    ('3to1at100000', 'independent'),
                                    ('3to1at300000', 'independent'),
                                    ('constant', 'backward'), ('constant', 'random-pass')]:
                settings = dict(replay_schedule=schedule, replay_order=order)
                with self.subTest(agent=agent, settings=settings):
                    self.assertEqual(registered_axes(dict(agent_settings={agent: settings}), agent),
                                     (schedule, order))

    def test_agents_without_explicit_axes_keep_their_default_path(self):
        for agent in (*UNMEASURED_AGENTS, 'acs2her_shortened', 'acs2er_episodic', *ARCHIVED_RUN_NAMES):
            with self.subTest(agent=agent):
                self.assertIsNone(registered_axes(dict(agent_settings={agent: {}}), agent))


if __name__ == '__main__':
    unittest.main()
