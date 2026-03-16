# Doped-HfO2

ABINIT input files, analysis scripts, and selected results for ferroelectric HfO2, including Y-,Sc- and Zr-doped systems.

## Scope
This repository contains:
- structural models for doped HfO2
- ABINIT SCF and Berry-phase input files
- Python scripts for polarization extraction and plotting
- figures

## Methods
- Density Functional Theory (DFT)
- Berry-phase polarization

## Systems
- Undoped HfO2
- Y-doped HfO2
- Sc-doped HfO2
- Zr-doped HfO
- selected oxygen-vacancy variants
- 
## Data Availability
Large ABINIT output files (wavefunctions `WFK.nc`, densities `DEN.nc`,
figures, and related outputs) are too large for GitHub and are archived on Zenodo:

DOI: https://doi.org/10.5281/zenodo.xxxxxxx

The Zenodo dataset contains:
- WFK files for ±P states
- DEN files for ±P states
- Additional large simulation data

This GitHub repository contains:
- ABINIT input files (`.in`)
- job scripts (`.slurm`)
- Python analysis and plotting scripts
- processed data
