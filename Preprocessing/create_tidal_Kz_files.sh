#!/bin/bash
# SGE Options
#$ -S /bin/bash
#$ -V
#$ -N Kz_files
#$ -M r.p.b.fischer@uu.nl
#$ -m e
#$ -l h_vmem=20G
#$ -l h_rt=2:00:00

echo 'Start'
cd ${HOME}/biofouling_3dtransport_2/Preprocessing/
python3 create_tidal_Kz_files.py
