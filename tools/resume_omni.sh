#!/usr/bin/env bash
# Resume the light-build speed run from its latest checkpoint if it isn't running.
# Usage: tools/resume_speed.sh [build] [final_iter]
set -euo pipefail
cd "$(dirname "$0")/.."
build="${1:-GL}"
final="${2:-1500}"
dir="out/ppoomni_${build}"
if [ -f "$dir/DONE" ]; then echo "finished (DONE marker)"; exit 0; fi
if pgrep -f "^./target/release/walker_lab ppoomni ${build}" >/dev/null; then
  echo "running: $(tail -1 "out/ppoomni_${build}.log" | cut -c1-60)"
  exit 0
fi
latest=$(ls "$dir"/policy_iter*.json | sort | tail -1)
iter=$(basename "$latest" .json | sed 's/policy_iter0*//')
vmax=$(python3 -c "import json,sys; l=json.load(open('$dir/log.json')); print([e['train'].get('vmax',3.0) for e in l if e['iter']==$iter][0])" 2>/dev/null || echo 3.0)
# Prefer the 10-iteration lightweight save if it is newer.
if [ -f "$dir/state.json" ]; then
  s_iter=$(python3 -c "import json; print(json.load(open('$dir/state.json'))['iter'])")
  if [ "$s_iter" -gt "$iter" ]; then
    cp "$dir/latest_policy.json" "$dir/resume_iter${s_iter}.json"
    latest="$dir/resume_iter${s_iter}.json"
    iter="$s_iter"
    vmax=$(python3 -c "import json; print(json.load(open('$dir/state.json'))['vmax'])")
  fi
fi
left=$(( final - iter ))
if [ "$left" -le 0 ]; then echo "finished at $iter"; exit 0; fi
n=$( (ls out/ppoomni_${build}_part*.log 2>/dev/null || true) | wc -l)
cp "out/ppoomni_${build}.log" "out/ppoomni_${build}_part$((n+1)).log"
cp "$dir/log.json" "$dir/log_part$((n+1)).json"
nohup ./target/release/walker_lab ppoomni "$build" "$left" "$latest" "$iter" > "out/ppoomni_${build}.log" 2>&1 &
echo "resumed from iter $iter at vmax $vmax ($left iterations left)"
