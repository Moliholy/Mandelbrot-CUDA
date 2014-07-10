#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH -A kurs2014-1-124

increment=500
n=500
maximum=10000

while [[ $n -le $maximum ]]
do

./a.out $n $n 0.0000022878 -0.7435669 .1314023 255 1 >>mandel.csv

n=$(( $n+$increment ))

done

