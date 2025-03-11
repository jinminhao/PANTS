#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate py39-app-vpn

SECONDS=0

BASEDIR=$(realpath "../")
echo $BASEDIR
LOGDIR=$BASEDIR/logs
mkdir -p $LOGDIR
LOGDIR=$BASEDIR/logs/adv_train
if [ -d $LOGDIR ]; then
  rm -rf $LOGDIR
fi
mkdir $LOGDIR

cd $BASEDIR/src/pants-app-end-host/attack


python3 attack_mlp.py -d adv-train-robust-mlp -f $LOGDIR/adv_train_mlp -a test -n 40
python3 attack_mlp.py -d pgd-train-robust-mlp -f $LOGDIR/pgd_train_mlp -a test -n 40
python3 attack_mlp.py -d amoeba-robust-mlp -f $LOGDIR/amoeba_robust_mlp -a test -n 40
python3 attack_mlp.py -d pants-robust-mlp -f $LOGDIR/pants_robust_mlp -a test -n 40

python3 attack_rf.py -d pgd-train-robust-rf -f $LOGDIR/pgd_train_rf -a test -n 40
python3 attack_rf.py -d amoeba-robust-rf -f $LOGDIR/amoeba_robust_rf -a test -n 40
python3 attack_rf.py -d pants-robust-rf -f $LOGDIR/pants_robust_rf -a test -n 40

python3 attack_tf.py -d adv-train-robust-tf -f $LOGDIR/adv_train_tf -a test -n 40
python3 attack_tf.py -d pgd-train-robust-tf -f $LOGDIR/pgd_train_tf -a test -n 40
python3 attack_tf.py -d amoeba-robust-tf -f $LOGDIR/amoeba_robust_tf -a test -n 40
python3 attack_tf.py -d pants-robust-tf -f $LOGDIR/pants_robust_tf -a test -n 40

python3 attack_cnn.py -d adv-train-robust-cnn -f $LOGDIR/adv_train_cnn -a test -n 40
python3 attack_cnn.py -d pgd-train-robust-cnn -f $LOGDIR/pgd_train_cnn -a test -n 40
python3 attack_cnn.py -d amoeba-robust-cnn -f $LOGDIR/amoeba_robust_cnn -a test -n 40
python3 attack_cnn.py -d pants-robust-cnn -f $LOGDIR/pants_robust_cnn -a test -n 40

cd $BASEDIR/plot
mkdir -p $BASEDIR/figures/
python3 plot_adv_train.py -f $BASEDIR/logs/adv_train --figure_dir $BASEDIR/figures/

duration=$SECONDS
# echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed." > adv-train-time.txt