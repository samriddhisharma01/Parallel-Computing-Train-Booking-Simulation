# Parallel-Computing-Train-Booking-Simulation

A train booking simulation with active bookings, cancellations and waitlist optimization. 

OpenMP used for parallel searching for seat availability. 

MPI is used for distributed waitlist resolution. Each rank is responsible for a subset of days. 


Simulation set-up:

Number of days: 30

Number of stations: 20

Number of seats: 200 

Total user requests: 100000

Computation time:
| MPI Nodes (-np) | OMP Threads | Runtime (sec) |
| :--- | :--- | :--- |
| **1** | **1** | **0.262 (Baseline)** |
| 1 | 2 | 2.783 |
| 1 | 4 | 7.844 |
| **2** | **1** | **0.204 (Best Performance)** |
| **4** | **1** | **0.208** |
| 2 | 2 | 2.718 |
| 2 | 4 | 9.026 |


For only 200 seats the OpenMP thread creation and synchronization overhead slows the computation for checking seat availibity.

For 100k requests MPI speeds up the waitlist processing when 2 ranks are used giving the fastest time for all combinations used. When 4 ranks are used the overhead again dominates the workload due to increased rank communication and reduced work per rank. 



