import json
from pathlib import Path

import networkx as nx
from gym_handeye.handeye_simulator import HandEyeSimulator
from gym_handeye.utils.utils import get_all_possible_transitions


def state(simulator):
    return [simulator.grip_pos_x, simulator.grip_pos_y,
            simulator.block_pos_x, simulator.block_pos_y,
            int(simulator.block_in_hand)]


def observation(side, gripper, block, held):
    result = ["w"] * (side * side) + ["0"]
    result[block] = "b"
    if held:
        result[-1] = "2"
    else:
        result[gripper] = "g"
        if gripper == block:
            result[-1] = "1"
    return result


def dump(side):
    simulator = HandEyeSimulator(side, True)
    states = []
    probes = []
    for gripper in range(side * side):
        for block in range(side * side):
            for held in ([False, True] if gripper == block else [False]):
                before = observation(side, gripper, block, held)
                simulator.parse_observation(before)
                start = state(simulator)
                states.append(start)
                for action in range(6):
                    simulator.parse_observation(before)
                    simulator.take_action(action)
                    probes.append([start, action, state(simulator),
                                   "".join(before), "".join(simulator.observe())])
    graph = nx.DiGraph()
    graph.add_nodes_from(tuple(start) for start in states)
    graph.add_edges_from((tuple(probe[0]), tuple(probe[2])) for probe in probes)
    distances = []
    for start in states:
        lengths = nx.single_source_shortest_path_length(graph, tuple(start))
        distances.append([
            min(length for end, length in lengths.items()
                if end[2:4] == (goal % side, goal // side))
            for goal in range(side * side)
        ])
    knowledge = sorted(["".join(before), action, "".join(after)]
                       for before, action, after in get_all_possible_transitions(side))
    print(f"g={side}: {len(states)} states, {len(probes)} probes, {len(knowledge)} knowledge transitions")
    return {"side": side, "states": states, "probes": probes,
            "knowledge": knowledge, "distances": distances}


def main():
    output = Path(__file__).resolve().parent.parent / "fixtures" / "hand_eye.json"
    output.write_text(json.dumps({"grids": [dump(side) for side in [3, 4, 5]]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
