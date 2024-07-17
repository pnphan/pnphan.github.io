---
permalink: /2.5-Layer-Base-Model/access_data/
title: "`access_data`"
author_profile: false
#redirect_from:
#  - /docs/
#  - /docs.html
---

[Back](./..)

`access_data` contains functions for creating and saving the output in netCDF4 files.

- `display_data` is a convenience function that prints all the metadata of the netCDF file i.e., information on the dimensions, attributes and variables.

- `create_file` creates a netCDF file on the disk. The parameters in the name list are saved as attributes. The variables `u1mat`, `u2mat`, `v1mat`, `v2mat`, `h1mat`, `h2mat`, `locsmat`, `ts` are created along with their own dimensions. These variables store `u1`, `u2`, `v1`, `v2`, `h1`, `h2`, `locs`, and time `t`.

- `last_timestep` takes a netCDF file and extracts the necessary data from the last timestep, to restart a previous simulation.

- `storedata` takes a variable and stores it in our netCDF file

- `storetime` does the same as above but for time `t`.

- `save_data` is just an amalgamation of `storedata` that calls `storedata` for `u1mat`, `u2mat`, `v1mat`, `v2mat`, `h1mat`, `h2mat`, `locsmat`, and `storetime` for the time `t`.
