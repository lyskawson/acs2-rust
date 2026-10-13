import json
from pathlib import Path

ARCHIVED_RUN_NAMES = json.loads(Path(__file__).with_suffix('.json').read_text())


def canonical_run_name(name):
    return ARCHIVED_RUN_NAMES.get(name, name)


def canonical_row(row):
    result = dict(row)
    if 'agent' in result:
        result['agent'] = canonical_run_name(result['agent'])
    return result


def agent_settings(config, agent):
    matches = [settings for name, settings in config.get('agent_settings', {}).items()
               if canonical_run_name(name) == canonical_run_name(agent)]
    if len(matches) > 1:
        raise ValueError('duplicate settings for an agent under archived and current names')
    return matches[0] if matches else {}
