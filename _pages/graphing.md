---
permalink: /2.5-Layer-Base-Model/graphing/
title: "`graphing`"
author_profile: false
#redirect_from:
#  - /docs/
#  - /docs.html
---

[Back](./..)

`graphing` contains functions for displaying and animating data.

- `animate` takes multple files (i.e., `files = ['data1.nc', 'data2.nc', ...]`) and produces a single smooth animation of a variable of your choosing using `matplotlib`. These choices are `u1`, `u2`, `v1`, `v2`, `h1`, `h2`, `zeta1`, and `zeta2` (vorticity). The function extracts the necessary grids (arrays) from the netCDF file, creates a figure and subplot, calculates the maximum and minimum values, and then plots a contour map with a colour bar. The animation runs through all the timesteps of your chosen variable.

- `view_slice` takes a single file and displays all 8 grids at a single timestep.
