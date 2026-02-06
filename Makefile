all:
	mpicc -fopenmp main.c -o train_sim

clean:
	rm -f train_sim
