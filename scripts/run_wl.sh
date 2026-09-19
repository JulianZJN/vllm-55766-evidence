#!/bin/bash
# usage: run_wl.sh <port> <label> [stop_after_nan=1] [seeds...]
PORT=$1; LABEL=$2; STOP=${3:-1}; shift 3
SEEDS=${@:-0 1 2 3 4 5 6 7 8 9}
PY=<VENV>/bin/python
extra=-1
for s in $SEEDS; do
  out=$($PY <WORKDIR>/control_probe.py --base http://127.0.0.1:$PORT --plen 16326 --max 200 --temp 0 --content natural --seed $s --label $LABEL 2>&1 | tail -1)
  echo "$(date +%T) $out"
  if [ "$STOP" = 1 ]; then
    if [ $extra -ge 0 ]; then extra=$((extra+1)); fi
    if echo "$out" | grep -q NAN && [ $extra -lt 0 ]; then extra=0; fi
    if [ $extra -ge 1 ]; then break; fi
  fi
done
echo WL_DONE
