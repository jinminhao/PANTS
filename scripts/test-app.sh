
BASEDIR=$(realpath "../")
echo $BASEDIR
LOGDIR=$BASEDIR/logs
mkdir -p $LOGDIR
LOGDIR=$BASEDIR/logs/app
if [ -d $LOGDIR ]; then
  rm -rf $LOGDIR
fi
mkdir $LOGDIR

cd $BASEDIR/src/pants-app-end-host/attack

python3 attack_mlp.py -d vanilla -f $LOGDIR/mlp_end_vanilla -a test -n 40
python3 attack_rf.py -d vanilla -f $LOGDIR/rf_end_vanilla -a test -n 40
python3 attack_tf.py -d vanilla -f $LOGDIR/tf_end_vanilla -a test -n 40
python3 attack_cnn.py -d vanilla -f $LOGDIR/cnn_end_vanilla -a test -n 40

python3 attack_mlp.py -d pants-robust-mlp -f $LOGDIR/mlp_end_pants -a test -n 40
python3 attack_rf.py -d pants-robust-rf -f $LOGDIR/rf_end_pants -a test -n 40
python3 attack_tf.py -d pants-robust-tf -f $LOGDIR/tf_end_pants -a test -n 40
python3 attack_cnn.py -d pants-robust-cnn -f $LOGDIR/cnn_end_pants -a test -n 40

python3 attack_mlp.py -d amoeba-robust-mlp -f $LOGDIR/mlp_end_amoeba -a test -n 40
python3 attack_rf.py -d amoeba-robust-rf -f $LOGDIR/rf_end_amoeba -a test -n 40
python3 attack_tf.py -d amoeba-robust-tf -f $LOGDIR/tf_end_amoeba -a test -n 40
python3 attack_cnn.py -d amoeba-robust-cnn -f $LOGDIR/cnn_end_amoeba -a test -n 40

cd $BASEDIR/src/pants-app-in-path/attack
python3 attack_mlp.py -d vanilla -f $LOGDIR/mlp_in_vanilla -a test -n 40
python3 attack_rf.py -d vanilla -f $LOGDIR/rf_in_vanilla -a test -n 40
python3 attack_tf.py -d vanilla -f $LOGDIR/tf_in_vanilla -a test -n 40
python3 attack_cnn.py -d vanilla -f $LOGDIR/cnn_in_vanilla -a test -n 40

python3 attack_mlp.py -d pants-robust-mlp -f $LOGDIR/mlp_in_pants -a test -n 40
python3 attack_rf.py -d pants-robust-rf -f $LOGDIR/rf_in_pants -a test -n 40
python3 attack_tf.py -d pants-robust-tf -f $LOGDIR/tf_in_pants -a test -n 40
python3 attack_cnn.py -d pants-robust-cnn -f $LOGDIR/cnn_in_pants -a test -n 40

python3 attack_mlp.py -d amoeba-robust-mlp -f $LOGDIR/mlp_in_amoeba -a test -n 40
python3 attack_rf.py -d amoeba-robust-rf -f $LOGDIR/rf_in_amoeba -a test -n 40
python3 attack_tf.py -d amoeba-robust-tf -f $LOGDIR/tf_in_amoeba -a test -n 40
python3 attack_cnn.py -d amoeba-robust-cnn -f $LOGDIR/cnn_in_amoeba -a test -n 40
