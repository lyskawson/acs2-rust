import json
from pathlib import Path

import networkx as nx
from gym.envs.toy_text.taxi import TaxiEnv


def main():
    env = TaxiEnv()
    probes = []
    graph = nx.DiGraph()
    for encoded, actions in env.P.items():
        before = list(env.decode(encoded))
        for action, outcomes in actions.items():
            assert len(outcomes) == 1
            probability, next_encoded, reward, done = outcomes[0]
            assert probability == 1.0
            after = list(env.decode(next_encoded))
            probes.append([before, action, after, reward, done])
            graph.add_edge(tuple(before[:3]), tuple(after[:3]))
    states = sorted(graph.nodes)
    distances = []
    for start in states:
        lengths = nx.single_source_shortest_path_length(graph, start)
        distances.append([min(length for end, length in lengths.items() if end[2] == goal) for goal in range(4)])
    initial = [list(env.decode(index)) for index, weight in enumerate(env.initial_state_distrib) if weight > 0]
    assert all(env.initial_state_distrib[env.encode(*state)] == 1 / 300 for state in initial)
    output = Path(__file__).resolve().parent.parent / "fixtures" / "taxi.json"
    output.write_text(json.dumps({"gym_version": "0.23.0", "probes": probes,
                                 "initial_states": initial, "states": states,
                                 "distances": distances}, indent=2) + "\n")
    print(f"{len(probes)} probes, {len(initial)} equally likely initial states, {len(states)} physical states")


if __name__ == "__main__":
    main()
