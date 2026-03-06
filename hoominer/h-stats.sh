#!/bin/bash
# Fetches Hoominer stats and formats for HiveOS

# Fetch stats from Hoominer API
stats_raw=$(curl -s http://127.0.0.1:8042/gpu)
if [ $? -ne 0 ]; then
    echo "Failed to fetch stats from Hoominer API"
    exit 1
fi

# Check if stats_raw is empty or invalid JSON
if [ -z "$stats_raw" ] || ! echo "$stats_raw" | jq . >/dev/null 2>&1; then
    echo "Invalid or empty API response"
    exit 1
fi

# Parse JSON using jq
khs=$(echo "$stats_raw" | jq '[.hash[] / 1000] | add // 0')
hs=$(echo "$stats_raw" | jq '[.hash[] / 1000] // []')
busid=$(echo "$stats_raw" | jq '[.busid[] | if . == "cpu" then 0 else . end] // []')
air=$(echo "$stats_raw" | jq '.air // 0')
accepted=$(echo "$stats_raw" | jq '.shares.accepted | add // 0')
rejected=$(echo "$stats_raw" | jq '.shares.rejected | add // 0')
hs_units="khs"
ver=$(echo "$stats_raw" | jq -r '.miner_version // "unknown"')
algo='hoohash'

# Calculate uptime
pid=$(pgrep -f hoominer | head -n1)
if [ -n "$pid" ]; then
    # Get uptime in seconds using ps
    uptime_seconds=$(ps -p "$pid" -o etimes= | tr -d '[:space:]')
    uptime=${uptime_seconds:-0}
else
    uptime="0"
fi

hs_length=$(echo "$hs" | jq 'length')
temp_length=$(echo "$temp" | jq 'length')
fan_length=$(echo "$fan" | jq 'length')

# Remove first element from temp and fan if hs array is shorter
if [ "$hs_length" -lt "$temp_length" ]; then
    temp=$(echo "$temp" | jq '.[1:] // .')
fi
if [ "$hs_length" -lt "$fan_length" ]; then
    fan=$(echo "$fan" | jq '.[1:] // .')
fi

# Format stats for HiveOS
stats=$(jq -n \
    --arg total_khs "$khs" \
    --argjson hs "$hs" \
    --arg hs_units "$hs_units" \
    --argjson temp "$temp" \
    --argjson fan "$fan" \
    --argjson bus_numbers "$busid" \
    --argjson accepted "$accepted" \
    --argjson rejected "$rejected" \
    --arg uptime "$uptime" \
    --arg ver "$ver" \
    --arg algo "$algo" \
    '{
        total_khs: $total_khs,
        hs: $hs,
        hs_units: $hs_units,
        temp: $temp,
        fan: $fan,
        bus_numbers: $bus_numbers,
        ar: [$accepted, $rejected],
        algo: $algo,
        uptime: $uptime,
        ver: $ver
    }')

echo "$stats"
