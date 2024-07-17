---
permalink: /2.5-Layer-Base-Model/
title: "2.5-Layer-Base-Model"
author_profile: false
#redirect_from:
#  - /docs/
#  - /docs.html
---

Work-in-progress documentation for the code of the 2.5-layer shallow water model for giant planet polar vortices by Morgan O'Neill, Daniel Stedman, and Peter Phan.

The new version of the code has the following general changes:

- Rewritten in Python from MATLAB
- New storm forcing function
- The Gaussians for the storm forcing are now generated on the entire domain instead of subdomains
- Data is saved to disk at every desired timestep, instead of storing it in memory until the program is finished running
- Parallelization of the code (first with `numba`, now with `MPI`)
- Added cumulus drag term
- Supports Jupiter-like parameters
- Supports netCDF4 output
- Supports initial conditions and restart files
- Various new functions for displaying data and animation

The code consists of the following main files:

- [`access_data`](./access_data) contains functions for creating and saving the output in netCDF4 files.
- [`graphing`](./graphing) contains functions for displaying and animating data.
- [`helper_functions_MPI`](./helper_functions_MPI) contains helper functions needed to run the simulation.
- `name_list_jupiter` contains Jupiter-like parameters for the model.
- `name_list` contains Saturn-like parameters for the model.
- [`run_model_MPI`](./run_model_MPI) is the main file that runs the simulation.
