# Doped-HfO2

ABINIT input files, analysis scripts, and selected results for ferroelectric HfO2, including Y-, Sc-, and Zr-doped systems.

## Scope
This repository contains:
- structural models for doped HfO2
- ABINIT SCF and Berry-phase input files
- Python scripts for polarization extraction and plotting

## Methods
- Density Functional Theory (DFT)
- Berry-phase polarization

## Systems
- Y-doped HfO2
- Sc-doped HfO2
- Zr-doped HfO2

## Computational Environment
The calculations were performed on a high-performance computing (HPC) system using parallel ABINIT simulations.

- Code: ABINIT
- Job scripts: SLURM (examples provided in `/slurm_jobs/`)

The workflow is fully portable and can be executed on any Linux-based HPC system with ABINIT installed.

## Reproducibility
To reproduce the calculations:

1. Use input files in `/abinit_inputs/`
2. Run SCF calculations: abinit < scf.in > scf.out
3. Run Berry-phase calculations: abinit < berry.in > berry.out
4. Analyze results using scripts in `/scripts/`

## Data Availability
Large ABINIT output files (wavefunctions `WFK.nc`, densities `DEN.nc`,
figures, and related outputs) are too large for GitHub and are archived on Zenodo:

- Sc doped DEN WFK and Paper Figures dataset (v1.0): https://doi.org/10.5281/zenodo.18988930  
- Y doped DEN WFK dataset (v1.0): https://doi.org/10.5281/zenodo.19043070  
- Zr doped DEN WFK dataset (v1.0): https://doi.org/10.5281/zenodo.19068910  

The Zenodo datasets contain:
- WFK files for ±P states  
- DEN files for ±P states  
- Figures  
- Additional large simulation data  

This GitHub repository contains:
- ABINIT input files (`.in`)
- job scripts (`.slurm`)
- Python analysis and plotting scripts
- processed data
