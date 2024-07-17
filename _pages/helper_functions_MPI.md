---
permalink: /2.5-Layer-Base-Model/helper_functions_MPI/
title: "`helper_functions_MPI`"
author_profile: false
#redirect_from:
#  - /docs/
#  - /docs.html
---

[Back](./..)

`helper_functions_MPI` contains various helper functions for running the simulation.

- `genlocs` generates a list of length 5 tuples. Each tuple corresponds to a single forced storm, and each contains an $x$-coordinate, a $y$-coordinate, a storm duration, a storm period, and a clock `tclock`.

- `pairshapeN2` takes the locations of the storms `locs`, and generates Gaussians at each location on the entire domain `wlayer`.

- `pairfieldN2(L, h1, wlayer)` takes `wlayer` and adds a correction term. It returns the actual storm forcing matrix used in the simulation, which corresponds to $S_{\text{st}}$ in the equations of motion.

- `viscND` calculates the viscosity terms. It corresponds to $-\text{Re}^{-1}\nabla^4\textbf{u}$ in the equations of motion.

- `BernN2` calculates the energy terms. It corresponds to $\nabla(\tilde{c_1^2}h_1 + \tilde{c_2^2}h_2 + \frac{1}{2}\Vert u_1 \Vert^2)$ or $\nabla(\gamma\tilde{c_1^2}h_1 + \tilde{c_2^2}h_2 + \frac{1}{2}\Vert u_1 \Vert^2)$ in the equations of motion.

- `xflux` and `yflux` correspond to $\nabla\cdot(\textbf{u}_ih_i)$ in the equations of motion.

- `calculate_KE` and `calculus_APE` return the kinetic energy and available potential energy.

- `split` splits the main domain into subdomains for parallelization. The subdomains are independently solved, and then combined using the `combine` function.

- `get_surrounding_points` is used to gather the boundary points of the subdomains.
