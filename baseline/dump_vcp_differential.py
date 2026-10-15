import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
from lcs import Perception
from lcs.agents.acs2 import ClassifiersList
from lcs.agents.acs2er import ReplayMemorySample
from lcs.agents.acs2her import Configuration
from lcs.agents.acs2vcp.ACS2VCPv10 import ACS2VCPv10

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'fixtures'))
from fixtures_common import cl_to_dict, dict_to_cl, sort_population
from dump_differential import gen_population_specs


def serialize(classifier):
    return dict(cl_to_dict(classifier), ee=classifier.ee)


def sample_dict(sample):
    return dict(state=list(sample.state), action=sample.action, reward=sample.reward,
                next_state=list(sample.next_state), done=sample.done)


def dump(count):
    source = Path(__import__('lcs').__file__).resolve().parents[2]
    cfg = Configuration(classifier_length=8, number_of_possible_actions=8, theta_i=0.0,
                        do_ga=False, do_pee=False, u_max=100000, er_samples_number=3)
    generation = random.Random(20261015)
    cases = []
    coverage = dict(heads={}, temperatures={}, rounds=0, learning_draws=0, priority_updates=0,
                    duplicate_samples=0, unchanged_priorities=0, no_change_maximum=0,
                    expected=0, unexpected=0, covering=0, deletions=0, rng_choice=0)
    for index in range(count):
        random.seed(generation.randrange(2 ** 32))
        heads = [1, 2, 4][index % 3]
        temperature = [0, 1, 2, 4][index % 4]
        agent = ACS2VCPv10(cfg, ensemble_size=heads, buffer_T=temperature)
        time = generation.randint(30, 500)
        samples = []
        for _ in range(generation.randint(3, 7)):
            state = [generation.choice('01') for _ in range(8)]
            following = [generation.choice('01') for _ in range(4)] + state[4:]
            samples.append(ReplayMemorySample(Perception(state), generation.randrange(8),
                           generation.choice([0.0, 1.0, 1000.0] if temperature <= 1 else [0.0, 1.0]), Perception(following),
                           generation.choice([False, True])))
        if index % 5 == 0:
            samples.append(samples[0])
            coverage['duplicate_samples'] += 1
        before = []
        for head in agent.ensemble_heads:
            chosen = generation.choice(samples)
            specs = gen_population_specs(list(chosen.state), list(chosen.next_state), chosen.action, time)
            head.population = ClassifiersList(*[dict_to_cl(spec, cfg) for spec in specs])
            before.append([serialize(cl) for cl in head.population])
            action_set = head.population.form_match_set(chosen.state).form_action_set(chosen.action)
            expected = sum(cl.does_anticipate_correctly(chosen.state, chosen.next_state) for cl in action_set)
            coverage['expected'] += int(expected > 0)
            coverage['unexpected'] += int(any(not cl.does_anticipate_correctly(chosen.state, chosen.next_state) for cl in action_set))
            coverage['covering'] += int(expected == 0)
            if action_set and not max(action_set, key=lambda cl: cl.fitness).does_anticipate_change():
                coverage['no_change_maximum'] += 1
        initial_maximum = generation.choice([0.0, 0.0, 3.0, 10.0])
        agent.memory.sigma2_max = initial_maximum
        agent.add_with_vcp(samples)
        priorities_before = list(agent.memory.priorities)
        sigmas_before = list(agent.memory.sigmas)
        maximum_before = agent.memory.sigma2_max
        indices = [[generation.randrange(len(samples)) for _ in range(cfg.er_samples_number)] for _ in range(heads)]
        sample_calls = []
        choices = []
        updates = []
        actual_numpy_choice = np.random.choice
        actual_choice = random.choice
        actual_update = agent.memory.update_priorities
        actual_remove = ClassifiersList.safe_remove

        def numpy_choice(size, batch_size, p):
            selected = indices[len(sample_calls)]
            assert size == len(samples) and batch_size == len(selected)
            sample_calls.append(p.tolist())
            return np.array(selected)

        def choice(sequence):
            position = actual_choice(range(len(sequence)))
            choices.append([len(sequence), position])
            return sequence[position]

        def update(selected, sigmas, priorities):
            updates.extend(dict(index=int(i), sigma=float(s), priority=float(p)) for i, s, p in zip(selected, sigmas, priorities))
            actual_update(selected, sigmas, priorities)

        def remove(self, item):
            coverage['deletions'] += 1
            return actual_remove(self, item)

        np.random.choice = numpy_choice
        random.choice = choice
        agent.memory.update_priorities = update
        ClassifiersList.safe_remove = remove
        try:
            agent.learn_new_buffer(time, {head: [None] for head in range(heads)})
        finally:
            np.random.choice = actual_numpy_choice
            random.choice = actual_choice
            ClassifiersList.safe_remove = actual_remove
        assert len(sample_calls) == heads
        assert all(probabilities == sample_calls[0] for probabilities in sample_calls)
        coverage['heads'][str(heads)] = coverage['heads'].get(str(heads), 0) + 1
        coverage['temperatures'][str(temperature)] = coverage['temperatures'].get(str(temperature), 0) + 1
        coverage['rounds'] += 1
        coverage['learning_draws'] += sum(map(len, indices))
        coverage['priority_updates'] += len(updates)
        coverage['unchanged_priorities'] += len(samples) - len(updates)
        coverage['rng_choice'] += len(choices)
        cases.append(dict(heads=heads, temperature=temperature, initial_maximum=initial_maximum,
                          samples=list(map(sample_dict, samples)), populations_before=before,
                          sigmas_before=sigmas_before, priorities_before=priorities_before,
                          maximum_before=maximum_before, probabilities=sample_calls[0],
                          learning_indices=indices, learning_draws=choices, time=time,
                          populations_after=[sort_population([serialize(cl) for cl in head.population]) for head in agent.ensemble_heads],
                          updates=updates, sigmas_after=agent.memory.sigmas,
                          priorities_after=agent.memory.priorities, maximum_after=agent.memory.sigma2_max))
    if coverage['deletions']:
        raise ValueError('pyalcs deletion skip hazard invalidates the gate')
    reference = Path(__import__('lcs.agents.acs2vcp', fromlist=['x']).__file__).parent
    return dict(reference_commit=subprocess.check_output(['/usr/bin/git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
                reference_sha256={name: hashlib.sha256((reference / name).read_bytes()).hexdigest()
                                  for name in ['ACS2VCPv10.py', 'PrioritizedReplayBuffer.py']},
                config=dict(beta=cfg.beta, gamma=cfg.gamma, theta_r=cfg.theta_r, theta_exp=cfg.theta_exp,
                            theta_as=cfg.theta_as, initial_q=cfg.initial_q), coverage=coverage, cases=cases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--cases', default=1000, type=int)
    args = parser.parse_args()
    result = dump(args.cases)
    args.out.write_text(json.dumps(result, separators=(',', ':')) + '\n')
    print(json.dumps(result['coverage']))
