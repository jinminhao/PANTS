#!/usr/bin/env bash

BASEDIR=$(realpath "../")
LOGDIR=$BASEDIR/logs
mkdir -p $LOGDIR

echo "Running test-adv-train.sh"
bash test-adv-train.sh

echo "Running test-asr.sh"
bash test-asr.sh

echo "Running test-important-features.sh"
bash test-important-features.sh

echo "Running test-more-threat.sh"
bash test-more-threat.sh

echo "Running test-netshare.sh"
bash test-netshare.sh

echo "Running test-robustification.sh"
bash test-robustification.sh

echo "Running test-transferability.sh"
bash test-transferability.sh
