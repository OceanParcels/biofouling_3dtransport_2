#!/bin/bash
# SGE Options
#$ -S /bin/bash
#$ -V
#$ -N regional_NPSG
#$ -M r.p.b.fischer@uu.nl
#$ -m e
#$ -l h_vmem=20G
#$ -l h_rt=18:00:00
#$ -q long.q

echo 'Start Simulation'
cd ${HOME}/biofouling_3dtransport_2/Simulation/
python3 regional_Kooi+NEMO_3D.py -mon=10 -yr=2003 -region='NPSG' -mixing='markov_0_KPP_ceiling_tides' -system='gemini' -bg_mixing='no' -diatom_death='NEMO' -no_advection='True'
