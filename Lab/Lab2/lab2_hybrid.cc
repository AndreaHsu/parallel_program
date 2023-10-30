#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	unsigned long long localSum = 0; 
	unsigned long long rr = r*r;
	int chunk_size = 64;
	
	int local_id, world_process_size; 
	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_process_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &local_id);


	#pragma omp parallel for schedule(guided, chunk_size) reduction(+:localSum)
		for (unsigned long long x = local_id; x < r; x+=world_process_size) {
			localSum += ceil(sqrtl(rr - x*x));
		}

	MPI_Reduce(&localSum, &pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

	if(local_id == 0){
		printf("%llu\n", (4 * (pixels % k)) % k);
	}

	MPI_Finalize();
	return 0;
}
