#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>

unsigned long long ncpus;
unsigned long long* thread_sum;
unsigned long long r;
unsigned long long k;
unsigned long long rr;

void* calculate(void *id){
	int tid = *(int*) id;
	unsigned long long *sum = new unsigned long long[1];
	*sum = 0;
	for (unsigned long long x = tid; x < r; x+=ncpus) {
		*sum += ceil(sqrtl(rr - x*x));
	}
	pthread_exit(sum);
}

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	r = atoll(argv[1]);
	k = atoll(argv[2]);
	rr = r*r;
	unsigned long long pixels = 0;
	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	ncpus = CPU_COUNT(&cpuset);
	pthread_t threads[ncpus];
	int* tid = new int[ncpus];
	// thread_sum = new unsigned long long[ncpus];

	for(int i = 0; i < ncpus; i++){
		tid[i] = i; 
		pthread_create(&threads[i], NULL, calculate, &tid[i]);
	}

	for(int i = 0; i < ncpus; i++){
		void* result;
		pthread_join(threads[i], &result);
		pixels += *(unsigned long long*)result;
	}

	printf("%llu\n", (4 * (pixels%k)) % k);
	pthread_exit(NULL);
	return 0;
}
