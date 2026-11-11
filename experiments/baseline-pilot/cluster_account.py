import getpass
import os
from pathlib import Path

DOTENV = Path(__file__).resolve().parents[2] / '.env'


def dotenv_values(path=DOTENV):
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, separator, value = line.removeprefix('export ').partition('=')
        if not separator:
            raise ValueError(f'malformed line in {path}: {line}')
        values[key.strip()] = value.strip().strip('\'"')
    return values


def setting(name):
    return os.environ.get(name) or dotenv_values().get(name)


def cluster_user():
    return setting('WCSS_USER') or getpass.getuser()


def required(name):
    value = setting(name)
    if not value:
        raise ValueError(f'{name} is not set: export it or copy .env.example to .env')
    return value


def grant_limit_seconds():
    return float(required('WCSS_GRANT_CPU_HOURS')) * 3600


def grant_usage_command():
    return ['sshare', '-U', '-u', cluster_user(), '-A', required('WCSS_ACCOUNT'), '-o', 'RawUsage', '-n']


def queue_command():
    return ['squeue', '--array', '-u', cluster_user(), '-h', '-o', '%i|%j|%t|%L|%C']
