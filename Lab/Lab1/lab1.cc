#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0; // local_sendbuf
	unsigned long long sum = 0; // local_recvbuf

	int local_id, world_process_size; 
	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_process_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &local_id);
	
	for (unsigned long long x = local_id; x < r; x += world_process_size) {
		unsigned long long y = ceil(sqrtl((r+x)*(r-x)));
		pixels += y;
		if (pixels >= k) pixels -= k;
		// printf("x: %llu, y: %llu, current_pixels: %d\n", x, y, pixels);
	}

	MPI_Reduce(&pixels, &sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

	if(local_id == 0){
		printf("%llu\n", (4 * sum) % k);
	}

	MPI_Finalize();
	return 0;
}
