import ast
import inspect
import json
import re
from pathlib import Path
from types import SimpleNamespace

import gym
import gym_maze.envs as environments
import numpy as np
import networkx as nx

from dump_maze_probes import dump_maze
from gym_maze.internal.maze_impl import MazeImpl
from gym_maze.utils.utils import get_all_possible_transitions


ROOT = Path(__file__).resolve().parent.parent
GEOMETRIES = ROOT / "acs2-envs" / "src" / "maze" / "geometries"


def dump_geometry(path):
    source = path.read_text()
    geometry_id = re.search(r'id: "([^"]+)"', source).group(1)
    name = geometry_id.removesuffix("-v0").removesuffix("-ounold")
    reference_name = "Woods101demi" if name == "Woods101_5" else name
    reference_id = reference_name + "-v0"
    reference_class = getattr(environments, reference_name)
    reference_initializable = True
    try:
        reference_matrix = reference_class().matrix
    except AssertionError:
        reference_initializable = False
        tree = ast.parse(inspect.getsource(reference_class))
        array_call = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "asarray"
        )
        reference_matrix = np.array(ast.literal_eval(array_call.args[0]))
    matrix = np.array([
        [int(value) for value in row.split(",") if value.strip()]
        for row in re.findall(r"&\[([0-9, ]+)\]", source)
    ])
    model = MazeImpl(matrix.copy())
    transitions = get_all_possible_transitions(SimpleNamespace(maze=model))
    walkable = sorted(tuple(cell) for cell in np.argwhere(matrix != 1))
    graph = nx.Graph()
    graph.add_nodes_from(walkable)
    graph.add_edges_from((start, end) for start, _, end in transitions)
    distances = []
    for start in walkable:
        lengths = nx.single_source_shortest_path_length(graph, start)
        distances.append([lengths.get(end) for end in walkable])
    probes = sorted([
        [[int(x) for x in start], action, [int(x) for x in end],
         "".join(model.perception(start)), "".join(model.perception(end))]
        for start, action, end in transitions
    ])
    matches = np.array_equal(matrix, reference_matrix)
    result = {
        "id": geometry_id,
        "reference_id": reference_id,
        "reference_max_episode_steps": gym.spec(reference_id).max_episode_steps,
        "matches_reference_grid": matches,
        "reference_initializable": reference_initializable,
        "grid": matrix.tolist(),
        "walkable_cells": [[int(value) for value in cell] for cell in walkable],
        "distances": distances,
        "transitions": probes,
    }
    if not matches:
        result["reference_grid"] = reference_matrix.astype(int).tolist()
    print(f"{geometry_id}: {len(probes)} transitions, reference grid equal={matches}")
    return result


def main():
    paths = sorted(GEOMETRIES.glob("*/*.rs"))
    entries = [dump_geometry(path) for path in paths if path.name != "mod.rs"]
    output = ROOT / "fixtures" / "maze_knowledge.json"
    output.write_text(json.dumps({"mazes": entries}, indent=2) + "\n")
    additions = [
        dump_maze(name + "-v0", getattr(environments, name))
        for name in ["Maze6", "MazeF3", "MazeB"]
    ]
    (ROOT / "fixtures" / "goal_maze_probes.json").write_text(
        json.dumps({"mazes": additions}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
