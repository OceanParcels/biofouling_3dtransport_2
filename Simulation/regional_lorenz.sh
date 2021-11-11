#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p normal 
#SBATCH -N 1 --ntasks-per-node=2
#SBATCH --job-name EqPac
#SBATCH --output EqPac
#SBATCH -o log.%j.o
#SBATCH -e log.%j.e
echo 'Initiating run'
srun python Simulation.py -mon='10' -yr='2003' -region='EqPac' -mixing='markov_0' -biofouling='MEDUSA' -rhobf='1170' -rhopl='920' -system='lorenz' -no_advection='False'
echo 'Finished computation.'