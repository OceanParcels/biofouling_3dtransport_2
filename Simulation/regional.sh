#!/bin/bash
#SBATCH -t 5:00:00
#SBATCH -N 1
#SBATCH --job-name GPGP122004
#SBATCH --output GPGP122004
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=r.p.b.fischer@uu.nl
cd $HOME/biofouling_3dtransport_2
srun python regional_Kooi+NEMO_3D.py -mon='12' -yr='2004' -region='GPGP' -no_biofouling=False -no_advection=False
