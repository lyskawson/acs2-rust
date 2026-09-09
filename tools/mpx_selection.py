ARM_COLUMNS = ("encoding", "epsilon", "u_max", "agent", "do_ga",
               "er_buffer_size", "er_min_samples", "er_samples_number")
RUN_COLUMNS = ("source", "block", "size", "seed", "repeat", "variant", *ARM_COLUMNS)


def arm_value(row, column):
    value = row.get(column, "")
    if column == "agent":
        return value or "acs2"
    if column.startswith("er_") and arm_value(row, "agent") != "acs2er":
        return ""
    if column == "epsilon" and value != "":
        return format(float(value), ".12g")
    return str(value)


def arm_of(row):
    return tuple(arm_value(row, column) for column in ARM_COLUMNS)


def run_key(row):
    return tuple(str(row.get(column, "")) for column in RUN_COLUMNS)


def selected(row, arm=None, sources=None):
    return (not sources or row["source"] in sources) and all(
        (str(row.get(column, "")) == str(value) if column.startswith("er_")
         else arm_value(row, column) == arm_value({**row, column: value}, column))
        for column, value in (arm or {}).items()
    )


def add_selection_arguments(parser):
    for column in ARM_COLUMNS:
        parser.add_argument("--" + column.replace("_", "-"), default=None)
    parser.add_argument("--block", default=None, help="one-based header block in a source log")
    parser.add_argument("--source", action="append", help="exact log basename; repeat to select multiple runs")


def selection_from(args):
    return {column: getattr(args, column) for column in (*ARM_COLUMNS, "block")
            if getattr(args, column) is not None}
