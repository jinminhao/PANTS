#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate py39-app-vpn

SECONDS=0

BASEDIR=$(realpath "../")
echo $BASEDIR
LOGDIR=$BASEDIR/logs
mkdir -p $LOGDIR
LOGDIR=$BASEDIR/logs/netshare
if [ -d $LOGDIR ]; then
  rm -rf $LOGDIR
fi
mkdir $LOGDIR

cd $BASEDIR/src/pants-app-end-host/attack

python3 attack_mlp.py -d netshare-robust-r1-mlp-x40 -f $LOGDIR/mlp_end_netshare_r1 -a test -n 40
python3 attack_mlp.py -d netshare-robust-r2-mlp-x59 -f $LOGDIR/mlp_end_netshare_r2 -a test -n 40
python3 attack_mlp.py -d netshare-robust-r3-mlp-x66 -f $LOGDIR/mlp_end_netshare_r3 -a test -n 40

python3 attack_rf.py -d netshare-robust-r1-rf-x42 -f $LOGDIR/rf_end_netshare_r1 -a test -n 40
python3 attack_rf.py -d netshare-robust-r2-rf-x44 -f $LOGDIR/rf_end_netshare_r2 -a test -n 40
python3 attack_rf.py -d netshare-robust-r3-rf-x46 -f $LOGDIR/rf_end_netshare_r3 -a test -n 40

python3 attack_cnn.py -d netshare-robust-r1-cnn-x40 -f $LOGDIR/cnn_end_netshare_r1 -a test -n 40
python3 attack_cnn.py -d netshare-robust-r2-cnn-x60 -f $LOGDIR/cnn_end_netshare_r2 -a test -n 40
python3 attack_cnn.py -d netshare-robust-r3-cnn-x73 -f $LOGDIR/cnn_end_netshare_r3 -a test -n 40

python3 attack_tf.py -d netshare-robust-r1-tf-x44 -f $LOGDIR/tf_end_netshare_r1 -a test -n 40
python3 attack_tf.py -d netshare-robust-r2-tf-x81 -f $LOGDIR/tf_end_netshare_r2 -a test -n 40
python3 attack_tf.py -d netshare-robust-r3-tf-x116 -f $LOGDIR/tf_end_netshare_r3 -a test -n 40
python3 attack_tf.py -d netshare-robust-r4-tf-x147 -f $LOGDIR/tf_end_netshare_r4 -a test -n 40
python3 attack_tf.py -d netshare-robust-r5-tf-x168 -f $LOGDIR/tf_end_netshare_r5 -a test -n 40

cd $BASEDIR/plot
mkdir -p $BASEDIR/figures/
python3 plot_netshare.py -f $LOGDIR --figure_dir $BASEDIR/figures/

duration=$SECONDS
# echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed." > netshare-time.txt
