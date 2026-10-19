#!/usr/bin/env bash
# Kaggle runs at most 2 GPU kernels at a time. Push the waiting ones as slots free up,
# each only once its input is ready:
#   audit-b       : no prerequisite
#   eft-baseline  : needs zoom-equivariance-eft finished (it reuses those checkpoints)
#   audit-a2      : rerun of audit-a with the same-pixel reference; after audit-a finished
cd "$(dirname "$0")/.." || exit 1
USER_K=rabeya100x

done_ok() { kaggle kernels status "$USER_K/zoom-equivariance-$1" 2>&1 | grep -q "COMPLETE"; }
try_push() { kaggle kernels push -p "kaggle/$1" 2>&1 | tee -a outputs/queue.log | grep -q "successfully pushed"; }

pending="build_audit_b build_eft_baseline build_audit_a2"
while [ -n "$pending" ]; do
    left=""
    for b in $pending; do
        ready=1
        [ "$b" = build_eft_baseline ] && ! done_ok eft && ready=0
        [ "$b" = build_audit_a2 ] && ! done_ok audit-a && ready=0
        if [ $ready = 1 ] && try_push "$b"; then
            echo "pushed $b"
        else
            left="$left $b"
        fi
    done
    pending=$(echo $left)
    [ -n "$pending" ] && sleep 240
done
echo "queue empty"
