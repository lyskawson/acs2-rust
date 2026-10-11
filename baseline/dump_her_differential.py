import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

from lcs import Perception
from lcs.agents.acs2 import ClassifiersList
from lcs.agents.acs2er import ReplayMemorySample
from lcs.agents.acs2her import ACS2HER, Configuration

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'fixtures'))
from fixtures_common import cl_to_dict, dict_to_cl, sort_population
from dump_differential import gen_population_specs


def serialize(classifier):
    return dict(cl_to_dict(classifier), ee=classifier.ee)


def dump(count):
    source = Path(__import__('lcs').__file__).resolve().parents[2]
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    cfg = Configuration(classifier_length=8, number_of_possible_actions=8, theta_i=0.0,
                        do_ga=False, do_pee=False, u_max=100000, er_samples_number=1)
    generation = random.Random(20261012)
    cases = []
    coverage = dict(original=0, relabeled=0, reached_copy=0, unreached_copy=0,
                    expected=0, unexpected=0, covering=0, rng_choice=0, deletions=0)
    for index in range(count):
        random.seed(generation.randrange(2 ** 32))
        length = generation.randint(1, 8)
        states = [[generation.choice('01') for _ in range(4)] for _ in range(length + 1)]
        real_goal = states[-1] if index % 3 == 0 else [generation.choice('01') for _ in range(4)]
        trial = [[Perception(states[t]), generation.randrange(8),
                  1000.0 if states[t + 1] == real_goal else 0.0,
                  Perception(states[t + 1]), t == length - 1] for t in range(length)]
        transition = generation.randrange(length)
        cfg.her_strategy = ['future', 'final', 'episode'][index % 3]
        cfg.her_goals_number = generation.randint(1, 3)
        agent = ACS2HER(cfg)
        goal_draws = []
        actual_sample = random.sample

        def sample(sequence, k):
            positions = actual_sample(range(len(sequence)), k)
            goal_draws.extend((len(sequence), position) for position in positions)
            return [sequence[position] for position in positions]

        random.sample = sample
        try:
            goals = agent.sample_goals(trial, transition)
        finally:
            random.sample = actual_sample
        state, action, reward, following, done = trial[transition]
        copies = [ReplayMemorySample(agent.state_goal_concat(state, Perception(real_goal)), action,
                                     reward, agent.state_goal_concat(following, Perception(real_goal)), done)]
        for goal in goals:
            copies.append(ReplayMemorySample(agent.state_goal_concat(state, goal), action,
                          agent.reward_function(following, goal), agent.state_goal_concat(following, goal), False))
        selected = 0 if index % 4 == 0 else generation.randrange(1, len(copies))
        experience = copies[selected]
        time = generation.randint(20, 500)
        steps = generation.randint(1, 18)
        specs = gen_population_specs(list(experience.state), list(experience.next_state), action, time)
        population = ClassifiersList(*[dict_to_cl(spec, cfg) for spec in specs])
        before = [serialize(cl) for cl in population]
        action_set = population.form_match_set(experience.state).form_action_set(action)
        expected = sum(cl.does_anticipate_correctly(experience.state, experience.next_state) for cl in action_set)
        coverage['expected'] += int(expected > 0)
        coverage['unexpected'] += int(any(not cl.does_anticipate_correctly(experience.state, experience.next_state) for cl in action_set))
        coverage['covering'] += int(expected == 0)
        coverage['original' if selected == 0 else 'relabeled'] += 1
        if selected > 0:
            coverage['reached_copy' if experience.reward == 1 else 'unreached_copy'] += 1
        agent.population = population
        agent.memory.add(experience)
        choices = []
        actual_choice = random.choice
        actual_remove = ClassifiersList.safe_remove

        def choice(sequence):
            position = actual_choice(range(len(sequence)))
            choices.append([len(sequence), position])
            return sequence[position]

        def remove(self, item):
            coverage['deletions'] += 1
            return actual_remove(self, item)

        random.choice = choice
        ClassifiersList.safe_remove = remove
        try:
            agent.learn(time, steps)
        finally:
            random.choice = actual_choice
            ClassifiersList.safe_remove = actual_remove
        coverage['rng_choice'] += len(choices)
        cases.append(dict(trial=[dict(state=list(t[0]), action=t[1], reward=t[2], next_state=list(t[3]), done=t[4]) for t in trial],
                          real_goal=real_goal, transition=transition, strategy=cfg.her_strategy, goals_number=cfg.her_goals_number,
                          goal_draws=goal_draws, samples=[dict(state=list(e.state), action=e.action, reward=e.reward,
                          next_state=list(e.next_state), done=e.done) for e in copies], selected=selected,
                          population_before=before, population_after=sort_population([serialize(cl) for cl in population]),
                          time=time, steps=steps, learning_draws=choices))
    if coverage['deletions']:
        raise ValueError('pyalcs deletion skip hazard invalidates the gate')
    files = ['ACS2HER.py', 'ReplayBuffer.py', 'Configuration.py']
    reference = Path(__import__('lcs.agents.acs2her', fromlist=['x']).__file__).parent
    return dict(reference_commit=commit, reference_sha256={name: hashlib.sha256((reference / name).read_bytes()).hexdigest() for name in files},
                coverage=coverage, config=dict(beta=cfg.beta, gamma=cfg.gamma, theta_i=cfg.theta_i, theta_r=cfg.theta_r,
                theta_exp=cfg.theta_exp, theta_as=cfg.theta_as, initial_q=cfg.initial_q, u_max=cfg.u_max), cases=cases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--cases', default=1000, type=int)
    args = parser.parse_args()
    result = dump(args.cases)
    args.out.write_text(json.dumps(result, separators=(',', ':')) + '\n')
    print(json.dumps(result['coverage']))
