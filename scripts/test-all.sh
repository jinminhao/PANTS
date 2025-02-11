#!/usr/bin/env bash

BASEDIR=$(realpath "../")
LOGDIR=$BASEDIR/logs
mkdir -p $LOGDIR

bash test-adv-train.sh
bash test-asr.sh
bash test-important-features.sh
bash test-more-threat.sh
bash test-netshare.sh
bash test-robustification.sh
bash test-transferability.sh
