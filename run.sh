EXAMPLES_PATH=/media/beni/ADAT/notesequences.tfrecord
OUTPUT_PATH=~/music_vae/run
CONFIG=attention-vae

mkdir -p $OUTPUT_PATH

python ~/PycharmProjects/magenta/magenta/models/music_vae/music_vae_train.py \
--config=$CONFIG \
--run_dir=$OUTPUT_PATH \
--mode=train \
--examples_path=$EXAMPLES_PATH