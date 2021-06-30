#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -p normal 
#SBATCH -N 1 --ntasks-per-node=2
#SBATCH --job-name NPSG
#SBATCH --output NPSG
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=r.p.b.fischer@uu.nl
echo 'Initiating run'
srun python Simulation.py -mon=10 -yr=2003 -region='NPSG' -mixing='markov_0' -biofouling='MEDUSA' -system='cartesius' -no_advection='True'
echo 'Finished computation.'
