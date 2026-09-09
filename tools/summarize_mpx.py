"""Render one multiplexer size as a table meant to be read, not parsed.

`mpx_verdicts.csv` is the machine-readable archive: 23 columns, every size, every
probe run that went nowhere. This turns one size into a Markdown page a person can
scan -- and it includes runs that are still in flight, which the verdict CSV cannot,
because a run only gets a verdict line when it stops.

Each row is one repeat. The state column distinguishes a finished run from a live
one: SUCCESS / TIME-LIMITED / MEMORY-LIMITED come from the verdict line, `running`
means trajectory points with no verdict yet, and `cancelled` / `partial` come from
the archive marker in the log name -- which appears as a suffix on some files and
infix on others, so it is matched anywhere. For everything but a verdict the numbers
are that run's last evaluation point, not a final result.

The four coverage columns are the diagnostic this project actually turns on: at
k=135 the wrong-answer classes (`a0_nc`, `a1_nc`) are what starve, so a run whose
knowledge sits at 0.75 or 0.50 is read there, not in the knowledge column.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

RUN_KEY = ("source", "seed", "repeat")
COVERAGE = ("a0_nochange", "a0_change", "a1_nochange", "a1_change")


def archived_state(source):
    """`cancelled` and `partial` mark a stopped run, as a suffix or an infix."""
    for marker in ("cancelled", "partial"):
        if f".{marker}" in source:
            return marker
    return "running"


def load(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def group_label(row):
    encoding = row["encoding"] or "?"
    epsilon = row["epsilon"] or "0.8"
    agent = row["agent"] or "acs2"
    name = "kanoniczne (flip)" if encoding == "flip" else f"zmienione ({encoding})"
    parts = [f"Kodowanie {name}", f"epsilon = {epsilon}"]
    if agent != "acs2":
        parts.append(agent)
    return " · ".join(parts)


def number(value, digits=4, dash="-"):
    if value in (None, ""):
        return dash
    try:
        return f"{float(value):.{digits}f}"
    except ValueError:
        return str(value)


def collect(trajectory, verdicts, size):
    """One record per repeat: its last trajectory point, plus its verdict if any."""
    last_point, records = {}, {}
    for row in trajectory:
        if int(row["size"]) != size:
            continue
        key = tuple(row[column] for column in RUN_KEY)
        previous = last_point.get(key)
        if previous is None or int(row["trials"]) >= int(previous["trials"]):
            last_point[key] = row

    for key, point in last_point.items():
        records[key] = {**point, "state": archived_state(point["source"]), "final": False}

    for row in verdicts:
        if int(row["size"]) != size:
            continue
        key = tuple(row[column] for column in RUN_KEY)
        merged = {**records.get(key, {}), **row}
        merged["state"] = row["verdict"]
        merged["final"] = True
        records[key] = merged
    return list(records.values())


def render(records, size):
    groups = defaultdict(list)
    for record in records:
        groups[group_label(record)].append(record)

    header = (
        "| ziarno | u_max | stan | próby | knowledge | accuracy | reguły | spec "
        "| a0_nc | a0_c | a1_nc | a1_c | godz. | prób/s | log |"
    )
    rule = "|" + "---|" * 15

    lines = [
        f"# MPX-{size} — wszystkie przebiegi",
        "",
        "Wygenerowane przez `tools/summarize_mpx.py` z `reports/mpx_verdicts.csv`",
        "i `reports/mpx_trajectory.csv`. **Nie edytować ręcznie** — przebudować po",
        "każdym ściągnięciu logów z klastra.",
        "",
        "Stan `running` znaczy, że przebieg nie ma jeszcze linii werdyktu, a `cancelled`",
        "albo `partial`, że został zatrzymany — w obu wypadkach liczby pochodzą",
        "z ostatniego punktu pomiarowego, nie z wyniku końcowego.",
        "",
        "`a0_nc` i `a1_nc` to klasy błędnej odpowiedzi. To one głodzą, więc sufit",
        "w kolumnie knowledge czyta się właśnie tam: dwie klasy puste dają 0,50,",
        "jedna 0,75. Pusta kolumna znaczy, że przebieg biegł bez `--log-coverage`.",
        "",
    ]

    def sort_group(label):
        return ("kanoniczne" not in label, label)

    for label in sorted(groups, key=sort_group):
        rows = sorted(
            groups[label],
            key=lambda r: (int(r["u_max"] or 0), int(r["seed"]), -int(r["trials"])),
        )
        lines += [f"## {label}", "", header, rule]
        for row in rows:
            wall = float(row["wall_s"]) / 3600 if row.get("wall_s") else 0.0
            rate = float(row["trials"]) / float(row["wall_s"]) if row.get("wall_s") else 0
            cells = [
                row["seed"],
                row["u_max"] or "-",
                f"**{row['state']}**" if row["state"] == "SUCCESS" else row["state"],
                f"{int(row['trials']):,}".replace(",", " "),
                number(row.get("knowledge")),
                number(row.get("accuracy")),
                row.get("reliable", "-"),
                number(row.get("spec"), 2),
                *(number(row.get(column), 4) for column in COVERAGE),
                f"{wall:.1f}",
                f"{rate:.0f}",
                f"`{row['source'].replace('slurm_mpx', '').replace('.out', '')}`",
            ]
            lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
        lines.append("")

    solved = [r for r in records if r["state"] == "SUCCESS"]
    lines += [
        "## Podsumowanie",
        "",
        f"- przebiegów w archiwum: **{len(records)}**",
        f"- rozwiązanych (knowledge = 1,0): **{len(solved)}**",
    ]
    if solved:
        def describe(row):
            trials = f"{int(row['trials']):,}".replace(",", " ")
            agent = row["agent"] or "acs2"
            return (
                f"ziarno {row['seed']}, {trials} prób, {float(row['wall_s']) / 3600:.1f} h, "
                f"kodowanie {row['encoding']}, epsilon {row['epsilon']}, {agent}"
            )

        fewest = min(solved, key=lambda r: int(r["trials"]))
        fastest = min(solved, key=lambda r: float(r["wall_s"]))
        lines.append(f"- najmniej prób: {describe(fewest)}")
        if fastest is not fewest:
            lines.append(f"- najkrótszy czas: {describe(fastest)}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=135)
    parser.add_argument("--trajectory-csv", type=Path, default=Path("reports/mpx_trajectory.csv"))
    parser.add_argument("--verdict-csv", type=Path, default=Path("reports/mpx_verdicts.csv"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    records = collect(load(args.trajectory_csv), load(args.verdict_csv), args.size)
    if not records:
        raise SystemExit(f"no runs at size {args.size}")
    out = args.out or Path(f"reports/MPX{args.size}_runs.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(records, args.size))
    print(f"{out}: {len(records)} runs")


if __name__ == "__main__":
    main()
