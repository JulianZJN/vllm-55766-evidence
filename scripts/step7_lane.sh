#!/bin/bash
# usage (LOGIN node): step7_lane.sh <jobid> <port> <serve_script> <build_label> <r...>
J=$1; PORT=$2; SERVE=$3; BUILD=$4; shift 4
D=<WORKDIR>; PY=<VENV>/bin/python
for r in "$@"; do
  SL=/tmp/step7_${BUILD}_r${r}.serve.log; WL=/tmp/step7_${BUILD}_r${r}.wl.log
  srun --jobid=$J --overlap --ntasks=1 bash $D/$SERVE > $SL 2>&1 &
  SRV=$!
  until grep -q "startup complete\|Traceback" $SL; do sleep 5; done
  NS="20"; case $r in 6n34|8n34) NS=34;; esac
  rr=${r%n34}
  srun --jobid=$J --overlap --ntasks=1 $PY $D/step7_ab.py --base http://127.0.0.1:$PORT --N $NS --r $rr --label ${BUILD} > $WL 2>&1
  srun --jobid=$J --overlap --ntasks=1 bash -c "pkill -u \$(id -u) -f \"api_server.*--port ${PORT: 0:3}[${PORT: -1}]\""
  wait $SRV 2>/dev/null
  sleep 8
done
echo LANE_DONE > /tmp/step7_${BUILD}_lane${J}.done
