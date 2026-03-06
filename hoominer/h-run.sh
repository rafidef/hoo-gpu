#!/bin/bash
# Runs Hoominer with configured parameters

. h-manifest.conf

mkdir -p "$(dirname "$CUSTOM_LOG_BASENAME.log")" && touch "$CUSTOM_LOG_BASENAME.log"

./hoominer $(< $CUSTOM_CONFIG_FILENAME) #--api-port ${CUSTOM_API_PORT} >> $CUSTOM_LOG_BASENAME.log 2>&1