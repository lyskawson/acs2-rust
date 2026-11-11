#!/bin/bash

load_dotenv() {
  local file="$1/.env" line key value
  [ -f "$file" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    case "$line" in '' | '#'*) continue ;; esac
    line="${line#export }"
    key="${line%%=*}"
    value="${line#*=}"
    if [ "$key" = "$line" ] || ! [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      echo "malformed line in $file: $line" >&2
      return 2
    fi
    case "$value" in
      \"*\" | \'*\') value="${value:1:${#value}-2}" ;;
    esac
    case "$value" in
      "~/"*) value="$HOME/${value:2}" ;;
    esac
    [ -n "${!key+x}" ] || export "$key=$value"
  done <"$file"
}

require_env() {
  local key missing=""
  for key in "$@"; do
    [ -n "${!key:-}" ] || missing="$missing $key"
  done
  if [ -n "$missing" ]; then
    echo "not set:$missing. Copy .env.example to .env in the repository root and fill it in." >&2
    return 2
  fi
}
