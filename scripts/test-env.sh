#!/usr/bin/env bash

BASEDIR=$(realpath "../")
LOGDIR=$BASEDIR/logs
mkdir -p $LOGDIR
LOGDIR=$BASEDIR/logs/test
if [ -d $LOGDIR ]; then
  rm -rf $LOGDIR
fi
mkdir $LOGDIR


eval "$(conda shell.bash hook)"

conda activate py39-app-vpn

cd $BASEDIR/src/pants-app-end-host/attack
python3 attack_mlp.py -d vanilla -f $LOGDIR/mlp_end_vanilla -a test -n 2

conda activate py39-vca
cd $BASEDIR/src/pants-vca-end-host/attack
python3 attack_tf.py -d vanilla -f $LOGDIR/tf_end_vanilla -a test -n 2
