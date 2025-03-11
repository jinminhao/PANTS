#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate py39-app-vpn

SECONDS=0

BASEDIR=$(realpath "../")
echo $BASEDIR
LOGDIR=$BASEDIR/logs
mkdir -p $LOGDIR
LOGDIR=$BASEDIR/logs/more-threat-models
if [ -d $LOGDIR ]; then
  rm -rf $LOGDIR
fi
mkdir $LOGDIR

cd $BASEDIR/src/more-threat-models/pants-app-end-host-custom/attack

python3 attack_mlp.py -d vanilla -f $LOGDIR/mlp_end_vanilla_d40_a40_i40_s0 -a test -n 40 --delay 40 --append 40 --inject 40
python3 attack_mlp.py -d pants-robust-mlp -f $LOGDIR/mlp_end_pants_d40_a40_i40_s0  -a test -n 40 --delay 40 --append 40 --inject 40

python3 attack_mlp.py -d vanilla -f $LOGDIR/mlp_end_vanilla_d39_a6_i8_s0  -a test -n 40 --delay 39 --append 6 --inject 8
python3 attack_mlp.py -d pants-robust-mlp -f $LOGDIR/mlp_end_pants_d39_a6_i8_s0  -a test -n 40 --delay 39 --append 6 --inject 8

python3 attack_mlp.py -d vanilla -f $LOGDIR/mlp_end_vanilla_d28_a19_i36_s0 -a test -n 40 --delay 28 --append 19 --inject 36
python3 attack_mlp.py -d pants-robust-mlp -f $LOGDIR/mlp_end_pants_d28_a19_i36_s0 -a test -n 40 --delay 28 --append 19 --inject 36

python3 attack_mlp.py -d vanilla -f $LOGDIR/mlp_end_vanilla_d34_a38_i19_s0 -a test -n 40 --delay 34 --append 38 --inject 19
python3 attack_mlp.py -d pants-robust-mlp -f $LOGDIR/mlp_end_pants_d34_a38_i19_s0 -a test -n 40 --delay 34 --append 38 --inject 19

python3 attack_mlp.py -d vanilla -f $LOGDIR/mlp_end_vanilla_d28_a25_i0_s0 -a test -n 40 --delay 28 --append 25 --inject 0
python3 attack_mlp.py -d pants-robust-mlp -f $LOGDIR/mlp_end_pants_d28_a25_i0_s0 -a test -n 40 --delay 28 --append 25 --inject 0

python3 attack_cnn.py -d vanilla -f $LOGDIR/cnn_end_vanilla_d40_a40_i40_s0 -a test -n 40 --delay 40 --append 40 --inject 40
python3 attack_cnn.py -d pants-robust-cnn -f $LOGDIR/cnn_end_pants_d40_a40_i40_s0  -a test -n 40 --delay 40 --append 40 --inject 40

python3 attack_cnn.py -d vanilla -f $LOGDIR/cnn_end_vanilla_d39_a6_i8_s0  -a test -n 40 --delay 39 --append 6 --inject 8
python3 attack_cnn.py -d pants-robust-cnn -f $LOGDIR/cnn_end_pants_d39_a6_i8_s0  -a test -n 40 --delay 39 --append 6 --inject 8

python3 attack_cnn.py -d vanilla -f $LOGDIR/cnn_end_vanilla_d28_a19_i36_s0 -a test -n 40 --delay 28 --append 19 --inject 36
python3 attack_cnn.py -d pants-robust-cnn -f $LOGDIR/cnn_end_pants_d28_a19_i36_s0 -a test -n 40 --delay 28 --append 19 --inject 36

python3 attack_cnn.py -d vanilla -f $LOGDIR/cnn_end_vanilla_d34_a38_i19_s0 -a test -n 40 --delay 34 --append 38 --inject 19
python3 attack_cnn.py -d pants-robust-cnn -f $LOGDIR/cnn_end_pants_d34_a38_i19_s0 -a test -n 40 --delay 34 --append 38 --inject 19

python3 attack_cnn.py -d vanilla -f $LOGDIR/cnn_end_vanilla_d28_a25_i0_s0 -a test -n 40 --delay 28 --append 25 --inject 0
python3 attack_cnn.py -d pants-robust-cnn -f $LOGDIR/cnn_end_pants_d28_a25_i0_s0 -a test -n 40 --delay 28 --append 25 --inject 0

cd $BASEDIR/src/more-threat-models/pants-app-end+-custom/attack

python3 attack_mlp.py -d vanilla -f $LOGDIR/mlp_end_vanilla_d40_a40_i40_s40 -a test -n 40 --delay 40 --append 40 --inject 40 --split 40
python3 attack_mlp.py -d pants-robust-mlp -f $LOGDIR/mlp_end_pants_d40_a40_i40_s40  -a test -n 40 --delay 40 --append 40 --inject 40 --split 40

python3 attack_mlp.py -d vanilla -f $LOGDIR/mlp_end_vanilla_d21_a5_i25_s15 -a test -n 40 --delay 21 --append 5 --inject 25 --split 15
python3 attack_mlp.py -d pants-robust-mlp -f $LOGDIR/mlp_end_pants_d21_a5_i25_s15  -a test -n 40 --delay 21 --append 5 --inject 25 --split 15

python3 attack_mlp.py -d vanilla -f $LOGDIR/mlp_end_vanilla_d16_a40_i19_s20 -a test -n 40 --delay 16 --append 40 --inject 19 --split 20
python3 attack_mlp.py -d pants-robust-mlp -f $LOGDIR/mlp_end_pants_d16_a40_i19_s20  -a test -n 40 --delay 16 --append 40 --inject 19 --split 20

python3 attack_cnn.py -d vanilla -f $LOGDIR/cnn_end_vanilla_d40_a40_i40_s40 -a test -n 40 --delay 40 --append 40 --inject 40 --split 40
python3 attack_cnn.py -d pants-robust-cnn -f $LOGDIR/cnn_end_pants_d40_a40_i40_s40  -a test -n 40 --delay 40 --append 40 --inject 40 --split 40

python3 attack_cnn.py -d vanilla -f $LOGDIR/cnn_end_vanilla_d21_a5_i25_s15 -a test -n 40 --delay 21 --append 5 --inject 25 --split 15
python3 attack_cnn.py -d pants-robust-cnn -f $LOGDIR/cnn_end_pants_d21_a5_i25_s15  -a test -n 40 --delay 21 --append 5 --inject 25 --split 15

python3 attack_cnn.py -d vanilla -f $LOGDIR/cnn_end_vanilla_d16_a40_i19_s20 -a test -n 40 --delay 16 --append 40 --inject 19 --split 20
python3 attack_cnn.py -d pants-robust-cnn -f $LOGDIR/cnn_end_pants_d16_a40_i19_s20  -a test -n 40 --delay 16 --append 40 --inject 19 --split 20

cd $BASEDIR/plot
mkdir -p $BASEDIR/figures/
python3 plot_more_threat.py -f $BASEDIR/logs/more-threat-models --figure_dir $BASEDIR/figures/


duration=$SECONDS
# echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed." > more-threat-time.txt
