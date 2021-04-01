#!/bin/bash
# SGE Options
#$ -S /bin/bash
#$ -V
#$ -N regional_EqPac
#$ -M r.p.b.fischer@uu.nl
#$ -m e
#$ -l h_vmem=20G
#$ -l h_rt=3:00:00

echo 'Start Simulation'
cd ${HOME}/biofouling_3dtransport_2/Simulation/
python3 regional_Kooi+NEMO_3D.py -mon=01 -yr=2004 -region='EqPac' -a_grazing='0.39' -mixing='markov_0_KPP_reflect' -system='gemini' -bg_mixing='tidal' -diatom_death='NEMO_detritus' -no_advection='True'
