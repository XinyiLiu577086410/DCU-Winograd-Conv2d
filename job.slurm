#!/bin/bash
#SBATCH -p wzidnormal
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=dcu:1
#SBATCH --mem=20G
#SBATCH -t 00:10:00
#SBATCH -o ./output/%j.out
#SBATCH -e ./output/%j.err

module list

module purge
module load compiler/dtk/24.04

module list

make clean
make

./winograd 64 256 14 14 256 3 3 1 1 0 0

