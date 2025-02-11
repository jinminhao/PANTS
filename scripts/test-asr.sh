#!/usr/bin/env bash
eval "$(conda shell.bash hook)"

SECONDS=0

BASEDIR=$(realpath "../")
echo $BASEDIR
LOGDIR=$BASEDIR/logs
mkdir -p $LOGDIR

conda activate py39-app-vpn

bash test-app.sh
bash test-vpn.sh

conda activate py39-vca

bash test-vca.sh

conda activate py39-app-vpn

LOGDIR=$BASEDIR/logs
cd $BASEDIR/plot
python3 plot_asr.py -f $LOGDIR --figure_dir /nfs/PANTS/figures/
python3 plot_robustified_asr.py -f $LOGDIR --figure_dir /nfs/PANTS/figures/

duration=$SECONDS
# echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed." > asr-time.txt