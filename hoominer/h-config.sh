#!/bin/bash
# Generates Hoominer configuration file

# Source HiveOS rig and wallet configurations
[[ -z $CUSTOM_TEMPLATE ]] && echo -e "${YELLOW}CUSTOM_TEMPLATE is empty${NOCOLOR}" && return 1
[[ -z $CUSTOM_URL ]] && echo -e "${YELLOW}CUSTOM_URL is empty${NOCOLOR}" && return 1

# Generate configuration
conf="--algorithm ${CUSTOM_ALGO} --user ${CUSTOM_TEMPLATE} --stratum ${CUSTOM_URL}"
[[ ! -z ${CUSTOM_PASS} ]] && conf+=" --pass ${CUSTOM_PASS} "
conf+=" ${CUSTOM_USER_CONFIG}"

[[ -z $CUSTOM_CONFIG_FILENAME ]] && echo -e "${RED}No CUSTOM_CONFIG_FILENAME is set${NOCOLOR}" && return 1
# Write configuration to file
echo -e "$conf" > $CUSTOM_CONFIG_FILENAME