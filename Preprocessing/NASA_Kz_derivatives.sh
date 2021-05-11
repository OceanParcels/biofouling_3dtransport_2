#!/bin/bash
# SGE Options
#$ -S /bin/bash
#$ -V
#$ -N dKz_NASA
#$ -M r.p.b.fischer@uu.nl
#$ -m e
#$ -l h_vmem=50G
#$ -l h_rt=8:00:00
#$ -q long.q

echo 'Start'
cd ${HOME}/biofouling_3dtransport_2/Preprocessing/
python3 NASA_Kz_derivatives.py
echo 'Finished'
