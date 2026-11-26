#!/bin/bash
# Run the same Locust flood against 1, 2 and 4 API containers.
# Usage: scripts/run_load_tests.sh [users] [duration]
set -euo pipefail

export PATH="$PATH:/usr/local/bin:/Applications/Docker.app/Contents/Resources/bin"
cd "$(dirname "$0")/.."

USERS="${1:-40}"
DURATION="${2:-60s}"
HOST="http://127.0.0.1:8080"
OUT="results/locust/${USERS}_users"
SPAWN_RATE=$(( USERS / 10 > 10 ? USERS / 10 : 10 ))
mkdir -p "$OUT"

for COUNT in 1 2 4; do
  echo "=== $COUNT container(s): $USERS users for $DURATION"
  docker compose up -d --wait --scale api="$COUNT" --remove-orphans
  docker compose exec -T gateway nginx -s reload

  # Warm up each container and confirm the load balancer reaches all of them.
  SEEN=$(for _ in $(seq 1 60); do
    curl -s -o /dev/null -D - -F "file=@demo/predict/demo_background.wav" "$HOST/predict" |
      awk 'tolower($1)=="x-container-id:" {print $2}' | tr -d '\r'
  done | sort -u | wc -l | tr -d ' ')
  echo "containers reached: $SEEN"
  if [ "$SEEN" -ne "$COUNT" ]; then
    echo "Load balancer reached $SEEN of $COUNT containers" >&2
    exit 1
  fi

  # Sample container CPU and memory while the flood is running.
  ( sleep 40; docker stats --no-stream --format '{{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}}' |
      grep mosquito-wingbeat-api > "$OUT/containers_${COUNT}_resources.txt" || true ) &
  STATS_PID=$!

  # --reset-stats leaves the ramp-up out of the reported numbers.
  .venv/bin/locust -f load_tests/locustfile.py --host "$HOST" --headless \
    --users "$USERS" --spawn-rate "$SPAWN_RATE" --reset-stats --run-time "$DURATION" --stop-timeout 70 \
    --csv "$OUT/containers_$COUNT" --html "$OUT/containers_$COUNT.html" --only-summary \
    2>&1 | grep -v -E 'NotOpenSSL|warnings.warn' | tail -32 || true
  wait "$STATS_PID" || true
  cat "$OUT/containers_${COUNT}_resources.txt"
done

.venv/bin/python scripts/summarize_load_tests.py "$OUT"
