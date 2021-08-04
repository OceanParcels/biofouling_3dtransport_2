#!/bin/bash
# SGE Options
#$ -S /bin/bash
#$ -V
#$ -N EqPac_rhobf1170
#$ -M d.m.a.lobelle@uu.nl
#$ -m e
#$ -l h_vmem=20G
#$ -l h_rt=18:00:00
#$ -q long.q

echo 'Start Simulation'
cd ${HOME}/biofouling_3dtransport_2/Simulation/
python3 Simulation.py -mon='01' -yr='2004' -region='EqPac' -mixing='markov_0' -system='gemini' -biofouling='MEDUSA' -rhobf='1170' -no_advection='True'
