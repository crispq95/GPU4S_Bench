#!/bin/bash
#
#SBATCH --job-name=cpq
#SBATCH -N 1 # number of nodes
#SBATCH --partition=bsc_cs

##SBATCH --gres=gpu:GeForceRTX3080:1

#module add cuda/11.2
#module add pgi/20.7
module load nvidia-hpc-sdk/22.3

#nvc -acc=cpu -ta=tesla -Minfo=all -o executable $1
#nvc -Minfo=all -o executable $1
#nvcc -arch=compute_60 -o executable $1 -lm

#if [ "$2" = "-prof" ]; then
#        echo "Nvidia profiler"
#        pgaccelinfo
#        nsys nvprof --print-gpu-trace ./executable
#else
#        perf stat ./executable
#fi
