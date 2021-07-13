#!/bin/bash
# SGE Options
#$ -S /bin/bash
#$ -V
#$ -N regional_SO
#$ -M d.m.a.lobelle@uu.nl
#$ -m e
#$ -l h_vmem=20G
#$ -l h_rt=18:00:00
#$ -q long.q

echo 'Start Simulation'
cd ${HOME}/biofouling_3dtransport_2/Simulation/
python3 Simulation.py -mon='10' -yr='2003' -region='SO' -mixing='markov_0' -system='gemini' -biofouling='MEDUSA' -no_advection='True'
