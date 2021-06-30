## Repository for biofouling study using NEMO-MEDUSA 2.0
This repository contains the code needed to produce the results for "Modeling submerged biofouled microplastics and their three-dimensional trajectories".
The study can be done in four steps: **Preprocessing**, **Simulation**, **Postprocessing** and **Analysis**.

### Preprocessing
In this step, vertical diffusivity fields are computed from global tidal mixing climatology (https://www.seanoe.org/data/00619/73082/)

### Simulation
The Lagrangian simulation of virtual particles representing microplastics. Bash scripts are used to run the script *Simulation.py*, which uses functions from *kernels.py* and *utils.py*

### Postprocessing
Computation of the individual oscillations from the particle trajectories. Computation of regional climatology for study locations.

### Analysis
Notebooks to explore the simulation. Scripts to create figures.
  