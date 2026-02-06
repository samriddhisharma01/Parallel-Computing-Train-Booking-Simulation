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
| 4 | 1 | 0.208 |
| 2 | 2 | 2.718 |
| 2 | 4 | 9.026 |
