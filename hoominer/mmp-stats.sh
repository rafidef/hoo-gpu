#!/bin/bash
GPU_COUNT=$1
LOG_FILE=$2
BASEDIR=$(dirname $0)
cd ${BASEDIR}

MINER_API_PORT=${MINER_API_PORT:-8042}
API_URL="http://127.0.0.1:${MINER_API_PORT}/gpu"

stats_json=$(curl --silent --insecure --header 'Accept: application/json' "$API_URL")
if [[ $? -ne 0 || -z $stats_json ]]; then
    echo "Miner API connection failed"
    exit 1
fi

echo "$stats_json"