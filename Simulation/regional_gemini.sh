#!/bin/bash
# SGE Options
#$ -S /bin/bash
#$ -V
#$ -N regional_SO
#$ -M r.p.b.fischer@uu.nl
#$ -m e
#$ -l h_vmem=30G
#$ -l h_rt=8:00:00
#$ -q long.q

echo 'Start Simulation'
cd ${HOME}/biofouling_3dtransport_2/Simulation/
python3 regional_Kooi+NEMO_3D.py -mon=01 -yr=2004 -region='SO' -dissolution='0.006' -mixing='markov_0_KPP_reflect' -grazing='full' -system='gemini' -bg_mixing='tidal' -diatom_death='NEMO_detritus' -no_advection='False'
