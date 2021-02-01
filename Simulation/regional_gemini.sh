#!/bin/bash
# SGE Options
#$ -S /bin/bash
#$ -V
#$ -N regional_EqPac
#$ -M r.p.b.fischer@uu.nl
#$ -m es
#$ -l h_vmem=20G
#$ -l h_rt=2:00:00

echo 'Start Simulation'
cd ${HOME}/biofouling_3dtransport_2/Simulation/
python3 regional_Kooi+NEMO_3D.py -mon=01 -yr=2004 -region='EqPac' -a_mort='0.16' -mixing='fixed' -system='gemini'
