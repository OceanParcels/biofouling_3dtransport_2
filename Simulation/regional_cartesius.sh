#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p normal 
#SBATCH -N 1 --ntasks-per-node=2
#SBATCH --job-name EqPac
#SBATCH --output EqPac
#SBATCH --mail-type=begin
#SBATCH --mail-type=end 
#SBATCH --mail-type=fail
#SBATCH --mail-user=d.m.a.lobelle@uu.nl
echo 'Initiating run'
srun python Simulation.py -mon='10' -yr='2002' -region='EqPac' -mixing='markov_0' -biofouling='MEDUSA' -rhobf='1170' -rhopl='920' -system='cartesius' -no_advection='True'
echo 'Finished computation.'