#!/bin/bash
# SGE Options
#$ -S /bin/bash
#$ -V
#$ -N climatology
#$ -M d.m.a.lobelle@uu.nl
#$ -m e
#$ -l h_vmem=20G
#$ -l h_rt=1:00:00
#$ -q long.q

echo 'Start Simulation'
cd ${HOME}/biofouling_3dtransport_2/Postprocessing/
python3 Climatology.py -region='EqPac'
python3 Climatology.py -region='NPSG'
python3 Climatology.py -region='SO'
