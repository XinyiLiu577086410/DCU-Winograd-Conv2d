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
# n c h w k r s u v p q
./winograd 256 192 14 14 192 3 3 1 1 0 0 

