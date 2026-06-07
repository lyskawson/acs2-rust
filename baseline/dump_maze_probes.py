"""Dump exhaustive maze transition/perception probes from the unmodified pyalcs
gym_maze environments, for differential testing against the Rust acs2-envs Maze.

For each maze and each path cell (matrix == MAZE_PATH), the agent is placed on
that cell and every one of the eight compass actions is executed once from a
fresh copy of the pristine matrix. Perception before and after, the scalar
reward, and the termination flag are recorded. The raw env class is used (no
gym.make TimeLimit wrapper), so 'done' reflects only reaching the reward cell;
truncation is verified separately on the Rust side. Probes are seedless and
exhaustive: positions are set explicitly, so no RNG enters the comparison.
"""

import json
from pathlib import Path

import numpy as np
import gym
import gym_maze  # noqa: F401  (import triggers gym_maze environment registration)

from gym_maze.common import MAZE_PATH, MAZE_ANIMAT
from gym_maze.internal.maze_impl import MazeImpl
from gym_maze.envs import Maze4, Maze5, Maze7, Woods1, Woods100

MAZES = [
    ("Maze4-v0", Maze4),
    ("Maze5-v0", Maze5),
    ("Maze7-v0", Maze7),
    ("Woods1-v0", Woods1),
    ("Woods100-v0", Woods100),
]


def dump_maze(maze_id, maze_class):
    max_episode_steps = gym.spec(maze_id).max_episode_steps
    env = maze_class()
    pristine = np.copy(env.matrix)
    path_cells = list(zip(*np.where(pristine == MAZE_PATH)))

    probes = []
    for row, col in path_cells:
        for action in range(8):
            env.maze = MazeImpl(np.copy(pristine))
            env.maze.matrix[row, col] = MAZE_ANIMAT
            perception_before = env.maze.perception()
            observation, reward, done, _ = env.step(action)
            probes.append({
                "row": int(row),
                "col": int(col),
                "action": action,
                "perception_before": perception_before,
                "perception_after": observation,
                "reward": int(reward),
                "done": bool(done),
            })

    return {
        "id": maze_id,
        "max_episode_steps": max_episode_steps,
        "grid": pristine.astype(int).tolist(),
        "probes": probes,
    }


def main():
    mazes = [dump_maze(*entry) for entry in MAZES]
    output_path = Path(__file__).resolve().parent.parent / "fixtures" / "maze_probes.json"
    output_path.write_text(json.dumps({"mazes": mazes}, indent=2))
    total = sum(len(maze["probes"]) for maze in mazes)
    print(f"wrote {output_path} ({len(mazes)} mazes, {total} probes)")


if __name__ == "__main__":
    main()
