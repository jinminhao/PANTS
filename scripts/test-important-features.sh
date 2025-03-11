#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate py39-app-vpn

SECONDS=0

BASEDIR=$(realpath "../")
echo $BASEDIR
LOGDIR=$BASEDIR/logs
mkdir -p $LOGDIR
LOGDIR=$BASEDIR/logs/important-features
if [ -d $LOGDIR ]; then
  rm -rf $LOGDIR
fi
mkdir $LOGDIR

cd $BASEDIR/src/important-features/pants-app-end-host-feat/attack
for ((i=2; i<=22; i+=2)); do
  python3 attack_mlp.py -d vanilla -f $LOGDIR/mlp_end_$i -a test -n 40 -k $i
done
for ((i=2; i<=22; i+=2)); do
  python3 attack_rf.py -d vanilla -f $LOGDIR/rf_end_$i -a test -n 40 -k $i
done

cd $BASEDIR/src/important-features/pants-app-in-path-feat/attack
for ((i=2; i<=22; i+=2)); do
  python3 attack_mlp.py -d vanilla -f $LOGDIR/mlp_in_$i -a test -n 40 -k $i
done
for ((i=2; i<=22; i+=2)); do
  python3 attack_rf.py -d vanilla -f $LOGDIR/rf_in_$i -a test -n 40 -k $i
done


cd $BASEDIR/plot
mkdir -p $BASEDIR/figures/
python3 plot_important_featres.py -f $LOGDIR --figure_dir $BASEDIR/figures/

duration=$SECONDS
# echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed." > important-feature-time.txt