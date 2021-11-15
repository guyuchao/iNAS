#!/usr/bin/env bash

model_dir=$1
model_name=$2
model_input_size=$3

adb push ${model_dir}/${model_name} /data/local/tmp
for i in `seq 1 10`
do
    echo "start run [${i}/10]"
    adb shell "/data/local/tmp/speed_benchmark_torch --model=/data/local/tmp/${model_name}" --input_dims=$model_input_size --input_type="float"
done
adb shell rm /data/local/tmp/${model_name}
