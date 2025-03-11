#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate py39-app-vpn

SECONDS=0

BASEDIR=$(realpath "../")
echo $BASEDIR
LOGDIR=$BASEDIR/logs
mkdir -p $LOGDIR
LOGDIR=$BASEDIR/logs/robustfication
if [ -d $LOGDIR ]; then
  rm -rf $LOGDIR
fi
mkdir $LOGDIR

cd $BASEDIR/src/pants-app-end-host/attack

python3 attack_mlp.py -d vanilla -f $LOGDIR/mlp_end_pants_r0 -a test -n 40
python3 attack_mlp.py -d pants-robust-mlp-r1 -f $LOGDIR/mlp_end_pants_r1 -a test -n 40
python3 attack_mlp.py -d pants-robust-mlp-r1 -f $LOGDIR/mlp_end_pants_r1 -a test -n 40
python3 attack_mlp.py -d pants-robust-mlp-r2 -f $LOGDIR/mlp_end_pants_r2 -a test -n 40
python3 attack_mlp.py -d pants-robust-mlp -f $LOGDIR/mlp_end_pants_r3 -a test -n 40

python3 attack_rf.py -d vanilla -f $LOGDIR/rf_end_pants_r0 -a test -n 40
python3 attack_rf.py -d pants-robust-rf-r1 -f $LOGDIR/rf_end_pants_r1 -a test -n 40
python3 attack_rf.py -d pants-robust-rf-r2 -f $LOGDIR/rf_end_pants_r2 -a test -n 40
python3 attack_rf.py -d pants-robust-rf -f $LOGDIR/rf_end_pants_r3 -a test -n 40

python3 attack_cnn.py -d vanilla -f $LOGDIR/cnn_end_pants_r0 -a test -n 40
python3 attack_cnn.py -d pants-robust-cnn-r1 -f $LOGDIR/cnn_end_pants_r1 -a test -n 40
python3 attack_cnn.py -d pants-robust-cnn-r2 -f $LOGDIR/cnn_end_pants_r2 -a test -n 40
python3 attack_cnn.py -d pants-robust-cnn -f $LOGDIR/cnn_end_pants_r3 -a test -n 40

python3 attack_tf.py -d vanilla -f $LOGDIR/tf_end_pants_r0 -a test -n 40
python3 attack_tf.py -d pants-robust-tf-r1 -f $LOGDIR/tf_end_pants_r1 -a test -n 40
python3 attack_tf.py -d pants-robust-tf-r2 -f $LOGDIR/tf_end_pants_r2 -a test -n 40
python3 attack_tf.py -d pants-robust-tf-r3 -f $LOGDIR/tf_end_pants_r3 -a test -n 40
python3 attack_tf.py -d pants-robust-tf-r4 -f $LOGDIR/tf_end_pants_r4 -a test -n 40
python3 attack_tf.py -d pants-robust-tf -f $LOGDIR/tf_end_pants_r5 -a test -n 40

cd $BASEDIR/plot
mkdir -p $BASEDIR/figures/
python3 plot_robustification.py -f $LOGDIR --figure_dir $BASEDIR/figures/

duration=$SECONDS
# echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed." > robustification-time.txt
