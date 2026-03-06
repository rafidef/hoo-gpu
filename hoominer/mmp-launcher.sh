#!/bin/bash

POOL=""
USER=""
PASSWORD=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c)
      shift 2
      ;;
    --pool)
      POOL="$2"
      shift 2
      ;;
    --user)
      USER="$2"
      shift 2
      ;;
    --password)
      PASSWORD="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! "$POOL" =~ ^stratum\+(tcp|ssl)://[^:]+:[0-9]+$ ]]; then
  echo "Error: Invalid --pool format. Must be stratum+tcp://host:port or stratum+ssl://host:port"
  exit 1
fi

CMD=("hoominer" "--stratum" "$POOL" "--user" "$USER" --disable-cpu)

if [[ -n "$PASSWORD" ]]; then
  CMD+=("--password" "$PASSWORD")
fi

CMD+=("${EXTRA_ARGS[@]}")
