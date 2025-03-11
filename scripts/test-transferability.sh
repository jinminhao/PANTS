#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate py39-app-vpn

SECONDS=0

BASEDIR=$(realpath "../")
LOGDIR=$BASEDIR/logs
mkdir -p $LOGDIR
LOGDIR=$BASEDIR/logs/trabsferability
if [ -d $LOGDIR ]; then
  rm -rf $LOGDIR
fi
mkdir $LOGDIR
cd $BASEDIR/src/transferability

python3 test_data_mlp_to_rf.py -f $BASEDIR/logs/trabsferability
python3 test_data_mlp_to_tf.py -f $BASEDIR/logs/trabsferability
python3 test_data_mlp_to_cnn.py -f $BASEDIR/logs/trabsferability

python3 test_data_rf_to_mlp.py -f $BASEDIR/logs/trabsferability
python3 test_data_rf_to_tf.py -f $BASEDIR/logs/trabsferability
python3 test_data_rf_to_cnn.py -f $BASEDIR/logs/trabsferability

python3 test_data_cnn_to_mlp.py -f $BASEDIR/logs/trabsferability
python3 test_data_cnn_to_rf.py -f $BASEDIR/logs/trabsferability
python3 test_data_cnn_to_tf.py -f $BASEDIR/logs/trabsferability

python3 test_data_tf_to_mlp.py -f $BASEDIR/logs/trabsferability
python3 test_data_tf_to_rf.py -f $BASEDIR/logs/trabsferability
python3 test_data_tf_to_cnn.py -f $BASEDIR/logs/trabsferability

cd $BASEDIR/plot
mkdir -p $BASEDIR/figures/
python3 plot_transferability.py -f $BASEDIR/logs/trabsferability --figure_dir $BASEDIR/figures/

duration=$SECONDS
# echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed." > transferability-time.txt
