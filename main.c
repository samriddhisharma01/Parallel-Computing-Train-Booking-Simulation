#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

#define DAYS 7
#define STATIONS 6
#define SEGMENTS (STATIONS - 1)
#define SEATS 10 

typedef struct {
    int day;
    int src;
    int dst;
} Request;

#define IDX(d, s, g) (((d) * SEATS * SEGMENTS) + ((s) * SEGMENTS) + (g))

/*  AVAILABILITY CHECK (OpenMP) */
int check_availability(const uint8_t* grid, int day, int src, int dst) {
    int available = 0;
    #pragma omp parallel for shared(available)
    for (int seat = 0; seat < SEATS; seat++) {
        if (available) continue;
        int free = 1;
        for (int seg = src; seg < dst; seg++) {
            if (grid[IDX(day, seat, seg)]) {
                free = 0;
                break;
            }
        }
        if (free) {
            #pragma omp atomic write
            available = 1;
        }
    }
    return available;
}

/*  SEAT ASSIGNMENT  */
int assign_seat(uint8_t* grid, int day, int src, int dst) {
    for (int seat = 0; seat < SEATS; seat++) {
        int free = 1;
        for (int seg = src; seg < dst; seg++) {
            if (grid[IDX(day, seat, seg)]) {
                free = 0;
                break;
            }
        }
        if (free) {
            for (int seg = src; seg < dst; seg++)
                grid[IDX(day, seat, seg)] = 1;
            return seat;
        }
    }
    return -1;
}

int main(int argc, char** argv) {
    int rank, size;
    int provided;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(NULL) + rank);

    if (rank == 0) {
        printf("==========================================\n");
        printf("TRAIN SIMULATION START | NODES: %d\n", size);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        printf("==========================================\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    uint8_t* occupied = (uint8_t*)calloc(DAYS * SEATS * SEGMENTS, sizeof(uint8_t));
    int total_reqs = 150; 
    Request* all_requests = (Request*)malloc(total_reqs * sizeof(Request));
    Request* waitlist = (Request*)malloc(total_reqs * sizeof(Request));
    int wl_count = 0;

    if (rank == 0) {

        // 1. Initial background bookings
        for (int i = 0; i < 50; i++) {
            assign_seat(occupied, rand() % DAYS, rand() % 3, 3 + rand() % 3);
        }

        // 2. Generate customer requests
        for (int i = 0; i < total_reqs; i++) {
            all_requests[i].day = rand() % DAYS;
            all_requests[i].src = rand() % (STATIONS - 1);
            all_requests[i].dst = all_requests[i].src + 1 +
                                   rand() % (STATIONS - all_requests[i].src - 1);
        }

        // 3. Process Live Bookings
        int live_confirmed = 0;
        for (int i = 0; i < total_reqs; i++) {
            if (check_availability(occupied,
                                   all_requests[i].day,
                                   all_requests[i].src,
                                   all_requests[i].dst)) {
                assign_seat(occupied,
                            all_requests[i].day,
                            all_requests[i].src,
                            all_requests[i].dst);
                live_confirmed++;
            } else {
                waitlist[wl_count++] = all_requests[i];
            }
        }
        printf("[Rank 0] Live phase: %d Confirmed, %d Waitlisted\n",
               live_confirmed, wl_count);

        // 4. CANCELLATION PHASE
        int cancelled = 0;
        for (int i = 0; i < DAYS * SEATS * SEGMENTS; i++) {
            if (occupied[i] && (rand() % 5 == 0)) {
                occupied[i] = 0;
                cancelled++;
            }
        }
        printf("[Rank 0] Cancellations: %d segments freed\n", cancelled);
    }

    // 5. SYNCHRONIZATION
    MPI_Bcast(&wl_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(waitlist, wl_count * sizeof(Request), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(occupied, DAYS * SEATS * SEGMENTS, MPI_UINT8_T, 0, MPI_COMM_WORLD);

    // 6. DISTRIBUTED WAITLIST RESOLUTION
    int local_cleared = 0;
    for (int i = 0; i < wl_count; i++) {
        if (waitlist[i].day % size == rank) {
            int seat = assign_seat(occupied,
                                   waitlist[i].day,
                                   waitlist[i].src,
                                   waitlist[i].dst);
            if (seat != -1) local_cleared++;
        }
    }

    printf("[Rank %d] processed its assigned days. Cleared %d from waitlist.\n",
           rank, local_cleared);

    // 7. FINAL MERGE
    uint8_t* global_grid =
        (uint8_t*)malloc(DAYS * SEATS * SEGMENTS * sizeof(uint8_t));

    MPI_Reduce(occupied, global_grid,
               DAYS * SEATS * SEGMENTS,
               MPI_UINT8_T, MPI_LOR,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    if (rank == 0) {
        int final_count = 0;
        for (int i = 0; i < DAYS * SEATS * SEGMENTS; i++)
            if (global_grid[i]) final_count++;

        printf("==========================================\n");
        printf("SIMULATION COMPLETE\n");
        printf("Final System Occupancy: %d segments\n", final_count);
        printf("Total Runtime: %.6f seconds\n", t_end - t_start);
        printf("==========================================\n");
    }

    free(occupied);
    free(all_requests);
    free(waitlist);
    free(global_grid);

    MPI_Finalize();
    return 0;
}





