#!/bin/bash
# SGE Options
#$ -S /bin/bash
#$ -V
#$ -N NASA_EqPac
#$ -M r.p.b.fischer@uu.nl
#$ -m e
#$ -l h_vmem=30G
#$ -l h_rt=8:00:00
#$ -q long.q

echo 'Start Simulation'
cd ${HOME}/biofouling_3dtransport_2/Simulation/
python3 regional_NASA.py
echo 'Done'
