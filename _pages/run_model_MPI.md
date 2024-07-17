---
permalink: /2.5-Layer-Base-Model/run_model_MPI/
title: "`run_model_MPI`"
author_profile: false
#redirect_from:
#  - /docs/
#  - /docs.html
---

[Back](./..)

`run_model_MPI` is the main file that runs the simulation. The code is parallelized for greater performance. Parallelizing programs requires a method for communication between the various steps and processes of the program. This is done using `MPI`, a standard protocol that defines this communication. The `mpi4py` library is the implementation of `MPI` for Python.

On the command line, `MPI` programs are run with `mpirun -n num python file_name.py` where `num` specifies the number of parallel workers. When running `MPI` programs on Slurm, MPI uses `--ntasks=num` to determine the number of parallel workers. The following code block specifies 65 parallel workers each with 2 CPUs. Every worker has a `rank`, numbered from `0` to `num-1`. Below, there are 65 workers numbered `0` to `64`.

```bash
#SBATCH --ntasks=65
#SBATCH --cpus-per-task=2
```

When running a program using `MPI`, the machine runs the program on each of the specified workers. For example, after we submit the script to run `run_model_MPI.py`, Niagara runs `run_model_MPI.py` on each of the 65 specified workers, with communication between them. This also means that if we wish for different workers to be running different tasks in the same file, our program must include statements like `if rank == k ...`

To utilize `MPI`, we must first tell the program to set up a communication world. It's also standard to define the `size` and `rank` variables. `size` returns the number of workers in the communication world. `rank` returns the rank of the current worker.

```python
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size() - 1
```

In `run_model_MPI.py`, we subtract 1 from `size` because we're going to use the rank 0 worker as a "root" worker, and the rank 1 to 64 workers will be the ones actually solving the equations on the grid. We'll now go through `run_model_MPI.py` line-by-line.

```python
def distribute_domain(domain_size, num_procs):
    subdomain_size = domain_size // np.sqrt(num_procs)
    remainder = domain_size % np.sqrt(num_procs)

    subdomains = [int(subdomain_size + 1) if i < remainder else int(subdomain_size) for i in range(num_procs)]
    return subdomains
```

The `distribute_domain` function takes as input the size of our domain `domain_size` i.e., the grid which we are solving on, and the number of workers `num_procs`. It returns a list of subdomain sizes. Note for this to work the number of threads used has to be a square number and the side length of the domain must have no remainder with the square root of the number of workers. This is easy to do if everything is in powers of 2, i.e `domain_size  = 1024` and `num_procs = 64`, then the domain size for each worker is 128. **Note: doesn't this make the remainder stuff redundant; if we're always making sure it's 0?**

The job of the rank 0 worker is to divide the grids `u1`, `u2`, `v1`, `v2`, `h1`, `h2` into evenly spaced grids among the other 64 workers. In the case of Jupiter, we have 64 grids of size 16 by 16, summing up to a bigger grid of size 1024 by 1024.

```python
subdomains = distribute_domain(N, size)
offset = subdomains[0]

subdomains = np.reshape(subdomains, (int(np.sqrt(size)),int(np.sqrt(size))))
ranks = np.reshape(np.arange(0,size), (int(np.sqrt(size)), int(np.sqrt(size)))) + 1

largeranks = np.zeros((int(3*np.sqrt(size)), int(3*np.sqrt(size))), dtype=ranks.dtype)
```

`subdomains` is first a 1 by 64 list of subdomain sidelengths. `offset` is the first element of this list. `subdomains` is then reshaped into an 8 by 8 array of subdomain sidelengths. `ranks` is an 8 by 8 array of the rank of each of the workers. `subdomains` and `ranks` are pictured below, respectively,

$$
\begin{bmatrix} 16 & 16 & 16 & 16 & 16 & 16 & 16 & 16 \\
                  16 & 16 & 16 & 16 & 16 & 16 & 16 & 16 \\
                  \vdots &&&&&&&\\
                  16 & 16 & 16 & 16 & 16 & 16 & 16 & 16 \\
\end{bmatrix}
\qquad
\qquad
\begin{bmatrix} 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 \\
                  9 & 10 & 11 & 12 & 13 & 14 & 15 & 16 \\
                  \vdots &&&&&&&\\
                  57 & 58 & 59 & 60 & 61 & 62 & 63 & 64 \\
\end{bmatrix}
$$

```python
largeranks = np.zeros((int(3*np.sqrt(size)), int(3*np.sqrt(size))), dtype=ranks.dtype)

for i in range(int(3*np.sqrt(size))):
    for j in range(int(3*np.sqrt(size))):
        largeranks[i, j] = ranks[i % int(np.sqrt(size)), j % int(np.sqrt(size))]
```

`largeranks` is first a 24 by 24 matrix of zeros. It's then modified by defining `largeranks[i, j] = ranks[i mod 8, j mod 8]`. It is now a 24 by 24 matrix, where each row corresponds to a row of `ranks` repeated 3 times. `largeranks` is pictured below

$$
\begin{bmatrix}
1 & \cdots & 8 & 1 & \cdots & 8 & 1 & \cdots & 8 \\
9 & \cdots & 16 & 9 & \cdots & 16 & 9 & \cdots & 16 \\
\vdots &&&&&&&&\\
57 & \cdots & 64 & 57 & \cdots & 64 & 57 & \cdots & 64
\end{bmatrix}
$$

```python
if rank == 0:
    #print(largeranks)
    if restart_name == None:
        if saving == True:
            ad.create_file(new_name)
        lasttime = 0
        locs = hf.genlocs(num, N, 0) ### use genlocs instead of paircount
    else:
        ad.create_file(new_name)
        u1, u2, v1, v2, h1, h2, locs, lasttime = ad.last_timestep(restart_name)
```

This block is only meant for the rank 0 worker. If we're starting a brand new simulation, the rank 0 worker creates a new file, generates new locations, and sets the time to `0`. If we're restarting from a previous simulation, it creates a new file and gathers all the data from the previous timestep to be used as the initial condition for the current run.

```python
if rank == 0:

    ...
    ...
    ...

    Wmat = None
    WmatSplit = None
    u1matSplit = [u1]
    v1matSplit = [v1]
    u2matSplit = [u2]
    v2matSplit = [v2]
    h1matSplit = [h1]
    h2matSplit = [h2]

    spdrag1Split = [spdrag1]
    spdrag2Split = [spdrag2]
    rdistSplit = [rdist]
    xSplit = [x]
    ySplit = [y]

    for i in range(1,size+1):
        #WmatSplit.append(hf.split(Wmat, offset, ranks, i))
        u1matSplit.append(hf.split(u1, offset, ranks, i))
        v1matSplit.append(hf.split(v1, offset, ranks, i))
        u2matSplit.append(hf.split(u2, offset, ranks, i))
        v2matSplit.append(hf.split(v2, offset, ranks, i))
        h1matSplit.append(hf.split(h1, offset, ranks, i))
        h2matSplit.append(hf.split(h2, offset, ranks, i))

        spdrag1Split.append(hf.split(spdrag1, offset, ranks, i))
        spdrag2Split.append(hf.split(spdrag2, offset, ranks, i))
        rdistSplit.append(hf.split(rdist, offset, ranks, i))
        xSplit.append(hf.split(x, offset, ranks, i))
        ySplit.append(hf.split(y, offset, ranks, i))
```

Each of the `Split` variables are lists of matrices that belong to each worker, ordered by rank. The rank 0 worker will have the main non-split matrix (i.e., the full 1024 by 1024 grid). This will be the first element in `Split`. The rank 1 worker will have the first subdomain matrix; this will be the second element in `Split`. The rank 2 worker will have the second subdomain matrix; this will be the third element in `Split`, and so on.

```python
else:
    WmatSplit = None
    u1matSplit = None
    v1matSplit = None
    u2matSplit = None
    v2matSplit = None
    h1matSplit = None
    h2matSplit = None

    spdrag1Split = None
    spdrag2Split = None
    rdistSplit = None
    xSplit = None
    ySplit = None

    u1 = None
    u2 = None
    v1 = None
    v2 = None
    h1 = None
    h2 = None
    Wmat = None
    wcorrect = None
    lasttime = None

    spdrag1 = None
    spdrag2 = None
    rdist = None
    x = None
    y = None
    locs = None
```

These variables are first set to `None` for the non-root workers, since they will be assigned to the non-root workers by the rank 0 worker at a later stage.

```python
u1 = comm.scatter(u1matSplit, root=0)
u2 = comm.scatter(u2matSplit, root=0)
v1 = comm.scatter(v1matSplit, root=0)
v2 = comm.scatter(v2matSplit, root=0)
h1 = comm.scatter(h1matSplit, root=0)
h2 = comm.scatter(h2matSplit, root=0)
spdrag1 = comm.scatter(spdrag1Split, root=0)
spdrag2 = comm.scatter(spdrag2Split, root=0)
rdist = comm.scatter(rdistSplit, root=0)
lasttime = comm.bcast(lasttime, root=0)
x = comm.scatter(xSplit, root=0)
y = comm.scatter(ySplit, root=0)
locs = comm.bcast(locs, root=0)
```

The `scatter` function takes `Split` and distributes contiguous sections of it across all the workers. Since `Split` is a list of 65 arrays, `scatter` assigns one array to each of the 65 workers. For example, the first array (the main non-split one) is assigned to the rank 0 worker. The second array (the first subdomain) is assigned to the rank 1 worker, and so on.

The `bcast` function distributes a copy of the argument to every worker. In this case, it distributes copies of `lasttime` and `locs` to every worker. The argument `root=0` specifies from which worker we are taking the data to be distributed from. In this case, we previously defined all this data in the rank 0 worker, and `root=0` tells the program to take this data from the rank 0 worker and distribute it.

```python
Wsum = None

if rank != 0:
    wlayer = hf.pairshapeN2(locs, 0, x, y, offset)
    Wsum = np.sum(wlayer) * dx**2

Wsums = comm.gather(Wsum, root=0)

if rank == 0:
    area = L**2
    wcorrect = np.sum(Wsums[1:]) / area

wcorrect = comm.bcast(wcorrect, root=0)

if rank != 0:
    Wmat = wlayer - wcorrect
```

This block initializes the storm forcing matrix. When `wlayer` is created by `pairshapeN2`, we still need to add the correction term. Each of the rank 1-64 workers creates a `Wsum` term. `gather` gathers `Wsum` from all the workers into a list `Wsums` and gives it to the rank 0 worker. The rank 0 worker computes the correction term `wcorrect`, and `bcast` sends a copy of `wcorrect` to every worker. The non-root workers then subtract the correction term from their `wlayer`.

```python
def timestep(u1,u2,v1,v2,h1,h2,Wmat, u1_p,u2_p,v1_p,v2_p,h1_p,h2_p):
```

The `timestep` function takes all data from the previous two timesteps and computes the data for the next timestep. The state variables are discretized along a doubly periodic Arakawa C grid, and the Adams-Bashforth method is the timestepping algorithm.

```python
    if AB == 2:
        tmp = u1.copy()
        u1 = 1.5 * u1 - 0.5 * u1_p
        u1_p = tmp  #
        tmp = u2.copy()
        u2 = 1.5 * u2 - 0.5 * u2_p
        u2_p = tmp  #
        tmp = v1.copy()
        v1 = 1.5 * v1 - 0.5 * v1_p
        v1_p = tmp
        tmp = v2.copy()
        v2 = 1.5 * v2 - 0.5 * v2_p
        v2_p = tmp
        tmp = h1.copy()
        h1 = 1.5 * h1 - 0.5 * h1_p
        h1_p = tmp
        if layers == 2.5:
            tmp = h2.copy()
            h2 = 1.5 * h2 - 0.5 * h2_p
            h2_p = tmp
```

The Adams-Bashforth method uses the following timestepping scheme,

$$ y*{n+2} = y*{n+1} + \Delta t\left(\frac{3}{2}f(y*{n+1}, t*{n+1}) - \frac{1}{2}f(y_n, t_n)\right), \qquad dy/dt = f $$

The `timestep` function computes `du1dt, du2dt, dv1dt, dv2dt, dh1dt, dh2dt`, multiplies it by $\Delta t$, or `dt` in the code. It then adds it to the previous timestep, just like above.

```python
    du1dt = hf.viscND(u1, Re, n)
    du2dt = hf.viscND(u2, Re, n)
    dv1dt = hf.viscND(v1, Re, n)
    dv2dt = hf.viscND(v2, Re, n)

    if spongedrag1 > 0:
        du1dt = du1dt - spdrag1 * (u1)
        du2dt = du2dt - spdrag2 * (u2)
        dv1dt = dv1dt - spdrag1 * (v1)
        dv2dt = dv2dt - spdrag2 * (v2)
```

`viscND` calculates the viscosity term. This corresponds to the term $\text{Re}^{-1}\nabla^4\textbf{u}$ in the equations of motion. The sponge layer is then added.

```python
    zeta1 = 1 - Bt * rdist**2 + (1 / dx) * (v1 - v1[:, l] + u1[l, :] - u1)

    zeta2 = 1 - Bt * rdist**2 + (1 / dx) * (v2 - v2[:, l] + u2[l, :] - u2)

    # add vorticity flux, zeta*u
    zv1 = zeta1 * (v1 + v1[:, l])
    zv2 = zeta2 * (v2 + v2[:, l])

    du1dt = du1dt + 0.25 * (zv1 + zv1[r, :])
    du2dt = du2dt + 0.25 * (zv2 + zv2[r, :])

    zu1 = zeta1 * (u1 + u1[l, :])
    zu2 = zeta2 * (u2 + u2[l, :])

    dv1dt = dv1dt - 0.25 * (zu1 + zu1[:, r])
    dv2dt = dv2dt - 0.25 * (zu2 + zu2[:, r])
```

The vorticity term is calculated and added. This corresponds to $-(1 - \tilde{\beta}\textbf{x}^2 + \zeta)\hat{\textbf{k}} \times \textbf{u}$ in the equations of motion. The `[:,l]`, `[:,r]`, `[l,:]`, `[r,:]` are used to compute finite differences. They shift the rows or columns of the matrix by 1, and have the same effect as `numpy.roll(u, 1, axis=1)`, `numpy.roll(u, -1, axis=1)`, `numpy.roll(u, 1, axis=0)`, `numpy.roll(u, -1, axis=0)`, respectively. This accounts for the doubly periodic domain.

```python
    du1dt = du1dt - (1 / dx) * u1 / dragf
    du2dt = du2dt - (1 / dx) * u2 / dragf
    dv1dt = dv1dt - (1 / dx) * v1 / dragf
    dv2dt = dv2dt - (1 / dx) * v2 / dragf

    B1p, B2p = hf.BernN2(u1, v1, u2, v2, gm, c22h, c12h, h1, h2, ord)

    du1dtsq = du1dt - (1 / dx) * (B1p - B1p[:, l])
    du2dtsq = du2dt - (1 / dx) * (B2p - B2p[:, l])

    dv1dtsq = dv1dt - (1 / dx) * (B1p - B1p[l, :])
    dv2dtsq = dv2dt - (1 / dx) * (B2p - B2p[l, :])
```

This block adds the cumulus drag term. `BernN2` then computes and adds the energy term, which corresponds to $\nabla(\tilde{c_1^2} h_1 + \tilde{c_2^2} h_2 + \frac{1}{2}\Vert\textbf{u}_1\Vert^2)$ or $\nabla(\gamma\tilde{c_1^2} h_1 + \tilde{c_2^2} h_2 + \frac{1}{2}\Vert\textbf{u}_2\Vert^2)$

```python
    if AB == 2:
        u1sq = u1_p + dt * du1dtsq
        u2sq = u2_p + dt * du2dtsq

        v1sq = v1_p + dt * dv1dtsq
        v2sq = v2_p + dt * dv2dtsq

    Fx1 = hf.xflux(h1, u1) - kappa / dx * (h1 - h1[:, l])
    Fy1 = hf.yflux(h1, v1) - kappa / dx * (h1 - h1[l, :])
    dh1dt = -(1 / dx) * (Fx1[:, r] - Fx1 + Fy1[r, :] - Fy1)

    if layers == 2.5:
        Fx2 = hf.xflux(h2, u2) - kappa / dx * (h2 - h2[:, l])
        Fy2 = hf.yflux(h2, v2) - kappa / dx * (h2 - h2[l, :])

        dh2dt = -(1 / dx) * (Fx2[:, r] - Fx2 + Fy2[r, :] - Fy2)

    if tradf > 0:
        dh1dt = dh1dt - 1 / tradf * (h1 - 1)
        dh2dt = dh2dt - 1 / tradf * (h2 - 1)

    if mode == 1:
        dh1dt = dh1dt + Wmat.astype(np.float64)
        if layers == 2.5:
            dh2dt = dh2dt - H1H2 * Wmat.astype(np.float64)

    if AB == 2:
        h1 = h1_p + dt * dh1dt
        if layers == 2.5:
            h2 = h2_p + dt * dh2dt

    u1 = u1sq
    u2 = u2sq
    v1 = v1sq
    v2 = v2sq
```

The Adams-Bashforth scheme is applied here to compute `u1`, `u2`, `v1`, `v2` for the next timestep. `xflux` and `yflux` correspond to the terms $\nabla \cdot (\textbf{u}_ih_i)$. `kappa / dx * (h - h[:, l])` corresponds to $\text{Pe}^{-1}\nabla^2h$. `1 / tradf * (h-1)` corresponds to $(h-1) / \tilde{t}$. The storm forcing matrix `Wmat` that's already computed is then added, and Adams-Bashforth is applied to compute `h1` and `h2`.

```python
    if math.isnan(h1[0, 0]):
        print(f"Rank: {rank}, h1 is nan")
        broke = True
```

This stops the timestepping if the matrix has a `nan` entry.

```python
mode = 1

# TIME STEPPING
if AB == 2:
    u1_p = u1.copy()
    v1_p = v1.copy()
    h1_p = h1.copy()
    u2_p = u2.copy()
    v2_p = v2.copy()
    h2_p = h2.copy()


ts = []
ii = 0

t = lasttime
tc = round(t / dt)

rem = False

tottimer = time.time()
# print("Starting simulation")

sendingTimes = []
simTimes = []
stormTimes = []
broke = False


initialmem = rss()
clocktimer = time.time()
```

This is the start of the simulation. **I think `sendingTimes`, `simTimes`, `stormTimes` is redundant?**

```python
while t <= tmax + lasttime + dt / 2:
    ### Running of the simulation on all ranks but the master rank (0) ###
    simtimer = time.time()

    if rank != 0:
        u1, u2, v1, v2, h1, h2, u1_p, u2_p, v1_p, v2_p, h1_p, h2_p, broke = timestep(
            u1, u2, v1, v2, h1, h2, Wmat, u1_p, u2_p, v1_p, v2_p, h1_p, h2_p
        )
```

The timestepping occurs in this loop. Only the non-root workers calculate the timestepping on their respective domains.

```python
    if broke == True:
        print(f"h1 Nan on rank {rank}")
        MPI.Finalize()
        MPI.COMM_WORLD.Abort()
```

This aborts the simulation if an entry is found to be `nan` (see earlier sections).

```python
    sendtimer = time.time()

    if rank != 0:
        ind = np.where(ranks == rank)
        i = ind[0][0] + int(np.sqrt(size))
        j = ind[1][0] + int(np.sqrt(size))
        sendranks, recvranks = hf.get_surrounding_points(largeranks, i, j)

        for sendrank in sendranks:
            if (sendrank[0], sendrank[1]) == (-1, -1):
                comm.send(
                    [
                        u1[2:4, :][:, 2:4],
                        u2[2:4, :][:, 2:4],
                        v1[2:4, :][:, 2:4],
                        v2[2:4, :][:, 2:4],
                        h1[2:4, :][:, 2:4],
                        h2[2:4, :][:, 2:4],
                    ],
                    dest=sendrank[2],
                    tag=0,
                )

            if (sendrank[0], sendrank[1]) == (-1, 0):
                comm.send(
                    [
                        u1[2:4, :][:, 2 : offset + 2],
                        u2[2:4, :][:, 2 : offset + 2],
                        v1[2:4, :][:, 2 : offset + 2],
                        v2[2:4, :][:, 2 : offset + 2],
                        h1[2:4, :][:, 2 : offset + 2],
                        h2[2:4, :][:, 2 : offset + 2],
                    ],
                    dest=sendrank[2],
                    tag=1,
                )

            if (sendrank[0], sendrank[1]) == (-1, 1):
                comm.send(
                    [
                        u1[2:4, :][:, offset : offset + 2],
                        u2[2:4, :][:, offset : offset + 2],
                        v1[2:4, :][:, offset : offset + 2],
                        v2[2:4, :][:, offset : offset + 2],
                        h1[2:4, :][:, offset : offset + 2],
                        h2[2:4, :][:, offset : offset + 2],
                    ],
                    dest=sendrank[2],
                    tag=2,
                )

            if (sendrank[0], sendrank[1]) == (0, -1):
                comm.send(
                    [
                        u1[2 : offset + 2, :][:, 2:4],
                        u2[2 : offset + 2, :][:, 2:4],
                        v1[2 : offset + 2, :][:, 2:4],
                        v2[2 : offset + 2, :][:, 2:4],
                        h1[2 : offset + 2, :][:, 2:4],
                        h2[2 : offset + 2, :][:, 2:4],
                    ],
                    dest=sendrank[2],
                    tag=3,
                )

            if (sendrank[0], sendrank[1]) == (0, 1):
                comm.send(
                    [
                        u1[2 : offset + 2, :][:, offset : offset + 2],
                        u2[2 : offset + 2, :][:, offset : offset + 2],
                        v1[2 : offset + 2, :][:, offset : offset + 2],
                        v2[2 : offset + 2, :][:, offset : offset + 2],
                        h1[2 : offset + 2, :][:, offset : offset + 2],
                        h2[2 : offset + 2, :][:, offset : offset + 2],
                    ],
                    dest=sendrank[2],
                    tag=4,
                )

            if (sendrank[0], sendrank[1]) == (1, -1):
                comm.send(
                    [
                        u1[offset : offset + 2, :][:, 2:4],
                        u2[offset : offset + 2, :][:, 2:4],
                        v1[offset : offset + 2, :][:, 2:4],
                        v2[offset : offset + 2, :][:, 2:4],
                        h1[offset : offset + 2, :][:, 2:4],
                        h2[offset : offset + 2, :][:, 2:4],
                    ],
                    dest=sendrank[2],
                    tag=5,
                )

            if (sendrank[0], sendrank[1]) == (1, 0):
                comm.send(
                    [
                        u1[offset : offset + 2, :][:, 2 : offset + 2],
                        u2[offset : offset + 2, :][:, 2 : offset + 2],
                        v1[offset : offset + 2, :][:, 2 : offset + 2],
                        v2[offset : offset + 2, :][:, 2 : offset + 2],
                        h1[offset : offset + 2, :][:, 2 : offset + 2],
                        h2[offset : offset + 2, :][:, 2 : offset + 2],
                    ],
                    dest=sendrank[2],
                    tag=6,
                )

            if (sendrank[0], sendrank[1]) == (1, 1):
                comm.send(
                    [
                        u1[offset : offset + 2, :][:, offset : offset + 2],
                        u2[offset : offset + 2, :][:, offset : offset + 2],
                        v1[offset : offset + 2, :][:, offset : offset + 2],
                        v2[offset : offset + 2, :][:, offset : offset + 2],
                        h1[offset : offset + 2, :][:, offset : offset + 2],
                        h2[offset : offset + 2, :][:, offset : offset + 2],
                    ],
                    dest=sendrank[2],
                    tag=7,
                )

        for sendrank in sendranks:
            if (sendrank[0], sendrank[1]) == (-1, -1):
                data = comm.recv(source=sendrank[2], tag=7)
                u1[0:2, :][:, 0:2] = data[0]
                u2[0:2, :][:, 0:2] = data[1]
                v1[0:2, :][:, 0:2] = data[2]
                v2[0:2, :][:, 0:2] = data[3]
                h1[0:2, :][:, 0:2] = data[4]
                h2[0:2, :][:, 0:2] = data[5]

            if (sendrank[0], sendrank[1]) == (-1, 0):
                data = comm.recv(source=sendrank[2], tag=6)
                u1[0:2, :][:, 2 : offset + 2] = data[0]
                u2[0:2, :][:, 2 : offset + 2] = data[1]
                v1[0:2, :][:, 2 : offset + 2] = data[2]
                v2[0:2, :][:, 2 : offset + 2] = data[3]
                h1[0:2, :][:, 2 : offset + 2] = data[4]
                h2[0:2, :][:, 2 : offset + 2] = data[5]

            if (sendrank[0], sendrank[1]) == (-1, 1):
                data = comm.recv(source=sendrank[2], tag=5)
                u1[0:2, :][:, offset + 2 : offset + 4] = data[0]
                u2[0:2, :][:, offset + 2 : offset + 4] = data[1]
                v1[0:2, :][:, offset + 2 : offset + 4] = data[2]
                v2[0:2, :][:, offset + 2 : offset + 4] = data[3]
                h1[0:2, :][:, offset + 2 : offset + 4] = data[4]
                h2[0:2, :][:, offset + 2 : offset + 4] = data[5]

            if (sendrank[0], sendrank[1]) == (0, -1):
                data = comm.recv(source=sendrank[2], tag=4)
                u1[2 : offset + 2, :][:, 0:2] = data[0]
                u2[2 : offset + 2, :][:, 0:2] = data[1]
                v1[2 : offset + 2, :][:, 0:2] = data[2]
                v2[2 : offset + 2, :][:, 0:2] = data[3]
                h1[2 : offset + 2, :][:, 0:2] = data[4]
                h2[2 : offset + 2, :][:, 0:2] = data[5]

            if (sendrank[0], sendrank[1]) == (0, 1):
                data = comm.recv(source=sendrank[2], tag=3)
                u1[2 : offset + 2, :][:, offset + 2 : offset + 4] = data[0]
                u2[2 : offset + 2, :][:, offset + 2 : offset + 4] = data[1]
                v1[2 : offset + 2, :][:, offset + 2 : offset + 4] = data[2]
                v2[2 : offset + 2, :][:, offset + 2 : offset + 4] = data[3]
                h1[2 : offset + 2, :][:, offset + 2 : offset + 4] = data[4]
                h2[2 : offset + 2, :][:, offset + 2 : offset + 4] = data[5]

            if (sendrank[0], sendrank[1]) == (1, -1):
                data = comm.recv(source=sendrank[2], tag=2)
                u1[offset + 2 : offset + 4, :][:, 0:2] = data[0]
                u2[offset + 2 : offset + 4, :][:, 0:2] = data[1]
                v1[offset + 2 : offset + 4, :][:, 0:2] = data[2]
                v2[offset + 2 : offset + 4, :][:, 0:2] = data[3]
                h1[offset + 2 : offset + 4, :][:, 0:2] = data[4]
                h2[offset + 2 : offset + 4, :][:, 0:2] = data[5]

            if (sendrank[0], sendrank[1]) == (1, 0):
                data = comm.recv(source=sendrank[2], tag=1)
                u1[offset + 2 : offset + 4, :][:, 2 : offset + 2] = data[0]
                u2[offset + 2 : offset + 4, :][:, 2 : offset + 2] = data[1]
                v1[offset + 2 : offset + 4, :][:, 2 : offset + 2] = data[2]
                v2[offset + 2 : offset + 4, :][:, 2 : offset + 2] = data[3]
                h1[offset + 2 : offset + 4, :][:, 2 : offset + 2] = data[4]
                h2[offset + 2 : offset + 4, :][:, 2 : offset + 2] = data[5]

            if (sendrank[0], sendrank[1]) == (1, 1):
                data = comm.recv(source=sendrank[2], tag=0)
                u1[offset + 2 : offset + 4, :][:, offset + 2 : offset + 4] = data[0]
                u2[offset + 2 : offset + 4, :][:, offset + 2 : offset + 4] = data[1]
                v1[offset + 2 : offset + 4, :][:, offset + 2 : offset + 4] = data[2]
                v2[offset + 2 : offset + 4, :][:, offset + 2 : offset + 4] = data[3]
                h1[offset + 2 : offset + 4, :][:, offset + 2 : offset + 4] = data[4]
                h2[offset + 2 : offset + 4, :][:, offset + 2 : offset + 4] = data[5]

    # sendingTimes.append(time.time()-sendtimer)

    ### Rank 0 checks for if new storms need to be created and sends out the new Wmat ###

    stormtimer = time.time()
```

In this block, the non-root workers will send their boundary conditions to other necessary workers. **`sendtimer` and `stormtimer` is redundant?**

```python
    if rank == 0:
        remove_layers = []  # store weather layers that need to be removed here
        rem = False

        if mode == 1:
            for i in range(len(locs)):
                if (t - locs[i][-1]) >= locs[i][3] and t != 0:
                    remove_layers.append(i)  # tag layer for removal if a storm's

            add = len(remove_layers)  # number of storms that were removed

            if add != 0:
                newlocs = hf.genlocs(add, N, t)

                for i in range(len(remove_layers)):
                    locs[remove_layers[i]] = newlocs[i]

                # wlayer = hf.pairshapeN2(locs, t) ### use pairshapeBEGIN instead of pairshape
                # Wmat = hf.pairfieldN2(L, h1, wlayer)

        if len(remove_layers) != 0:
            rem = True

            # WmatSplit = [Wmat]
            # for i in range(1,size+1):
            #    WmatSplit.append(hf.split(Wmat, offset, ranks, i))

    rem = comm.bcast(rem, root=0)
    locs = comm.bcast(locs, root=0)
    if rem == True:
        if rank != 0:
            wlayer = hf.pairshapeN2(locs, 0, x, y, offset)
            # Wmat = hf.pairfieldN2(L, h1, wlayer)
            Wsum = np.sum(wlayer) * dx**2

        Wsums = comm.gather(Wsum, root=0)

        if rank == 0:
            area = L**2
            wcorrect = np.sum(Wsums[1:]) / area

        wcorrect = comm.bcast(wcorrect, root=0)

        if rank != 0:
            Wmat = wlayer - wcorrect

        rem = False
```

This is the storm forcing function. The rank 0 worker checks if new storms need to be created, and computes and sends out the `Wmat` if so, in the same manner as before.

```python
    if tc % tpl == 0 and saving == True:
        ### Combining data on rank 0 ###
        u1matSplit = comm.gather(u1, root=0)
        v1matSplit = comm.gather(v1, root=0)
        u2matSplit = comm.gather(u2, root=0)
        v2matSplit = comm.gather(v2, root=0)
        h1matSplit = comm.gather(h1, root=0)
        h2matSplit = comm.gather(h2, root=0)

        if rank == 0:
            u1 = hf.combine(u1matSplit, offset, ranks, size)
            u2 = hf.combine(u2matSplit, offset, ranks, size)
            v1 = hf.combine(v1matSplit, offset, ranks, size)
            v2 = hf.combine(v2matSplit, offset, ranks, size)
            h1 = hf.combine(h1matSplit, offset, ranks, size)
            h2 = hf.combine(h2matSplit, offset, ranks, size)

            print(f"t={t}, time elapsed {time.time()-clocktimer}")

            ad.save_data(u1, u2, v1, v2, h1, h2, locs, t, lasttime, new_name)
```

For every `tpl` number of timesteps, the non-root workers send `u1`, `u2`, `v1`, `v2`. `h1`, `h2` to the rank 0 worker. The rank 0 worker then combines the subdomains into a single large domain using the `combine` function, and the data is saved to a netCDF file using the `save_data` function.

```python
    tc += 1
    t = tc * dt
```

The number of timesteps is increased by 1, and the new raw time is computed. The loop repeats until `t` reaches `tmax`.

[//]: # <kbd>MPI</kbd>
[//]: # `python`
