import hashlib
import json
from pathlib import Path

from plot_mpx import apply_style, load, numeric, plot_anatomy, plot_reach, plot_signal


def main():
    inputs = {name: Path(f"reports/mpx_{name}.csv")
              for name in ("trajectory", "diagnostics", "verdicts")}
    trajectory = numeric([row for row in load(inputs["trajectory"]) if row["knowledge"] != ""],
                         "trials", "wall_s", "knowledge", "reliable", "spec", "pop")
    verdicts = numeric(load(inputs["verdicts"]), "trials", "knowledge", "reliable", "spec", "wall_s")
    diagnostics = numeric(load(inputs["diagnostics"]), "trials", "addr_spec", "addr_random", "addr_full")
    out = Path("reports/figures")
    formats = ["pdf", "png"]
    canonical70 = ["slurm_mpx70_s42_addr.out", *[f"slurm_mpx70_s{seed}.out" for seed in range(43, 47)]]
    recipes = []
    apply_style()
    for suffix in ("", "_canonical"):
        arm = {"encoding": "flip", "epsilon": "0.8", "u_max": "8", "agent": "acs2"}
        plot_reach(trajectory, verdicts, 70, "pyalcs", out, formats, arm, suffix, canonical70)
        recipes.append({"figure": "reach", "size": 70, "suffix": suffix, "sources": canonical70, "arm": arm})
    for seed, source in ((42, canonical70[0]), (46, canonical70[-1])):
        plot_anatomy(trajectory, 70, "pyalcs", seed, out, formats, sources=[source])
        recipes.append({"figure": "anatomy", "size": 70, "seed": seed, "sources": [source]})
    for suffix, encoding, epsilon, sources in (
        ("_canonical_eps1", "flip", "1", ["slurm_mpx135_s42_eps1_u11.out", "slurm_mpx135_s43_eps1_u11.out"]),
        ("_outcome_u11", "outcome", "0.8", [f"slurm_mpx135_s{seed}_outcome_u11.out" for seed in (42, 43, 45, 46)]
         + ["slurm_mpx135_s44_outcome_u11.cancelled"]),
    ):
        arm = {"encoding": encoding, "epsilon": epsilon, "u_max": "11", "agent": "acs2"}
        plot_reach(trajectory, verdicts, 135, "pyalcs", out, formats, arm, suffix, sources)
        recipes.append({"figure": "reach", "size": 135, "suffix": suffix, "sources": sources, "arm": arm})
    sources = ["slurm_mpx70_s42_addr.out", "slurm_mpx135_s42_addr.out"]
    plot_signal(diagnostics, {70, 135}, out, formats, sources=sources)
    recipes.append({"figure": "signal", "sizes": [70, 135], "seed": 42, "sources": sources})
    manifest = {"variant": "pyalcs", "recipes": recipes,
                "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs.values()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
