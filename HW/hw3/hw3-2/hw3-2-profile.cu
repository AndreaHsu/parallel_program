#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

//======================
#define DEV_NO 0
cudaDeviceProp prop;
const int INF = ((1 << 30) - 1);
const int Blocksize = 64;
const int Half = 32;
int* Dist;
int realn, n, m;

int ceil(int a, int b) { return (a + b - 1) / b; }
void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    
    realn = n;
    n = ceil(n, Blocksize) * Blocksize;
    Dist = (int*) malloc(n * n * sizeof(int));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i * n + j] = 0;
            } else {
                Dist[i * n + j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0] * n + pair[1]] = pair[2];
    }
    
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < realn; ++i) {
        fwrite(Dist + i * n, sizeof(int), realn, outfile);
    }
    
    fclose(outfile);
}

__global__ void block_FW_p1(int* dist, int round, int n){
    __shared__ int shr[Blocksize][Blocksize];
    int x = threadIdx.x; // col
    int y = threadIdx.y; // row

    int c = round * Blocksize + threadIdx.x;
    int r = round * Blocksize + threadIdx.y;

    shr[y][x] = dist[r * n + c];
    shr[y + Half][x] = dist[(r + Half) * n + c];
    shr[y][x + Half] = dist[r * n + (c + Half)];
    shr[y + Half][x + Half] = dist[(r + Half) * n + (c + Half)];

    __syncthreads();

    for(int i = 0; i < Blocksize; i++){
        shr[y][x] = min(shr[y][x], shr[y][i] + shr[i][x]);
        shr[y + Half][x] = min(shr[y + Half][x], shr[y + Half][i] + shr[i][x]);
        shr[y][x + Half] = min(shr[y][x + Half], shr[y][i] + shr[i][x + Half]);
        shr[y + Half][x + Half] = min(shr[y + Half][x + Half], shr[y + Half][i] + shr[i][x + Half]);
        __syncthreads();
    }
    
    dist[r * n + c] = shr[y][x];
    dist[(r + Half) * n + c] = shr[y + Half][x];
    dist[r * n + (c + Half)] = shr[y][x + Half];
    dist[(r + Half) * n + (c + Half)] = shr[y + Half][x + Half];
    return;
}

__global__ void block_FW_p2(int* dist, int round, int n){
    if(blockIdx.y == round) return;
    __shared__ int shr[Blocksize][Blocksize];
    __shared__ int row[Blocksize][Blocksize];
    __shared__ int col[Blocksize][Blocksize];
    // I am a thread in the block beside the current pivot block
    int x = threadIdx.x; // col
    int y = threadIdx.y; // row
    // A thread in the current pivot block
    int pivotc = round * Blocksize + threadIdx.x; // col
    int pivotr = round * Blocksize + threadIdx.y; // row
    // The responsible position for me
    int respc = blockIdx.y * Blocksize + threadIdx.x; // if the responsible row is the same as pivotr, calculate the responsible col
    int respr = blockIdx.y * Blocksize + threadIdx.y; // if the responsible col is the same as pivotc, calculate the responsible row
    
    // load pivot
    shr[y][x] = dist[pivotr * n + pivotc];
    shr[y + Half][x] = dist[(pivotr + Half) * n + pivotc];
    shr[y][x + Half] = dist[pivotr * n + (pivotc + Half)];
    shr[y + Half][x + Half] = dist[(pivotr + Half) * n + (pivotc + Half)];

    // load the same row as pivot
    row[y][x] = dist[pivotr * n + respc]; 
    row[y + Half][x] = dist[(pivotr + Half) * n + respc]; 
    row[y][x + Half] = dist[pivotr * n + (respc + Half)]; 
    row[y + Half][x + Half] = dist[(pivotr + Half) * n + (respc + Half)]; 

    // load the same col as pivot
    col[y][x] = dist[respr * n + pivotc];
    col[y + Half][x] = dist[(respr + Half) * n + pivotc];
    col[y][x + Half] = dist[respr * n + (pivotc + Half)];
    col[y + Half][x + Half] = dist[(respr + Half) * n + (pivotc + Half)];

    __syncthreads();

    #pragma unroll 32
    for(int i = 0; i < Blocksize; i++){
        row[y][x] = min(row[y][x], shr[y][i] + row[i][x]);
        row[y + Half][x] = min(row[y + Half][x], shr[y + Half][i] + row[i][x]);
        row[y][x + Half] = min(row[y][x + Half], shr[y][i] + row[i][x + Half]);
        row[y + Half][x + Half] = min(row[y + Half][x + Half], shr[y + Half][i] + row[i][x + Half]);

        col[y][x] = min(col[y][x], col[y][i] + shr[i][x]);
        col[y + Half][x] = min(col[y + Half][x], col[y + Half][i] + shr[i][x]);
        col[y][x + Half] = min(col[y][x + Half], col[y][i] + shr[i][x + Half]);
        col[y + Half][x + Half] = min(col[y + Half][x + Half], col[y + Half][i] + shr[i][x + Half]);
        __syncthreads();
    }
    
    dist[pivotr * n + respc] = row[y][x]; 
    dist[(pivotr + Half) * n + respc] = row[y + Half][x]; 
    dist[pivotr * n + (respc + Half)] = row[y][x + Half]; 
    dist[(pivotr + Half) * n + (respc + Half)] = row[y + Half][x + Half]; 

    dist[respr * n + pivotc] = col[y][x];
    dist[(respr + Half) * n + pivotc] = col[y + Half][x];
    dist[respr * n + (pivotc + Half)] = col[y][x + Half];
    dist[(respr + Half) * n + (pivotc + Half)] = col[y + Half][x + Half];
    return;
}

__global__ void block_FW_p3(int* dist, int round, int n){
    if(blockIdx.x == round || blockIdx.y == round) return;
    __shared__ int shr[Blocksize][Blocksize];
    __shared__ int row[Blocksize][Blocksize];
    __shared__ int col[Blocksize][Blocksize];
    // I am the thread in the block beside the phase2 blocks
    int x = threadIdx.x; // col
    int y = threadIdx.y; // row
    // my real col and real row in the whole matrix
    int realc = blockIdx.x * Blocksize + threadIdx.x;
    int realr = blockIdx.y * Blocksize + threadIdx.y;
    // The needed position to calculate my value
    int neededc = round * Blocksize + threadIdx.x; // if the needed row is the same as me, calculate the needed col
    int neededr = round * Blocksize + threadIdx.y; // if the needed col is the same as me, calculate the needed col

    // load the same row as me
    row[y][x] = dist[realr * n + neededc]; 
    row[y + Half][x] = dist[(realr + Half) * n + neededc]; 
    row[y][x + Half] = dist[realr * n + (neededc + Half)]; 
    row[y + Half][x + Half] = dist[(realr + Half) * n + (neededc + Half)]; 

    // load the same column as me
    col[y][x] = dist[neededr * n + realc];
    col[y + Half][x] = dist[(neededr + Half) * n + realc];
    col[y][x + Half] = dist[neededr * n + (realc + Half)];
    col[y + Half][x + Half] = dist[(neededr + Half) * n + (realc + Half)];
    __syncthreads();

    shr[y][x] = dist[realr * n + realc];
    shr[y + Half][x] = dist[(realr + Half) * n + realc];
    shr[y][x + Half] = dist[realr * n + (realc + Half)];
    shr[y + Half][x + Half] = dist[(realr + Half) * n + (realc + Half)];

    #pragma unroll 32
    for(int i = 0; i < Blocksize; i++){
        shr[y][x] = min(shr[y][x], row[y][i] + col[i][x]);
        shr[y + Half][x] = min(shr[y + Half][x], row[y + Half][i] + col[i][x]);
        shr[y][x + Half] = min(shr[y][x + Half], row[y][i] + col[i][x + Half]);
        shr[y + Half][x + Half] = min(shr[y + Half][x + Half], row[y + Half][i] + col[i][x + Half]);
    }
    
    dist[realr * n + realc] = shr[y][x];
    dist[(realr + Half) * n + realc] = shr[y + Half][x];
    dist[realr * n + (realc + Half)] = shr[y][x + Half];
    dist[(realr + Half) * n + (realc + Half)] = shr[y + Half][x + Half];
    return;
}


int main(int argc, char* argv[]) {
    struct timespec io_instart, io_inend, io_outstart, io_outend;
    double io_elapsed = 0;
    clock_gettime(CLOCK_MONOTONIC, &io_instart);
    input(argv[1]);
    clock_gettime(CLOCK_MONOTONIC, &io_inend);
    io_elapsed += (io_inend.tv_sec - io_instart.tv_sec) + (io_inend.tv_nsec - io_instart.tv_nsec) / 1e9;
    int* ddist;
    // cudaHostRegister(Dist, n * n * sizeof(int), cudaHostRegisterDefault);
    cudaMalloc(&ddist, n * n * sizeof(int));
    cudaMemcpy(ddist, Dist, n * n * sizeof(int), cudaMemcpyHostToDevice);

    // cudaGetDeviceProperties(&prop, DEV_NO);
    // printf("maxThreasPerBlock = %d, sharedMemPerBlock = %d", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);
    // maxThreasPerBlock = 1024, sharedMemPerBlock = 49152
    int B = n / Blocksize;
    dim3 num_blocks_p1(1, 1);
    dim3 num_blocks_p2(1, B);
    dim3 num_blocks_p3(B, B);
    dim3 num_threads(Half, Half);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for(int i = 0; i < B; i++){
        block_FW_p1<<<num_blocks_p1, num_threads>>>(ddist, i, n);
        block_FW_p2<<<num_blocks_p2, num_threads>>>(ddist, i, n);
        block_FW_p3<<<num_blocks_p3, num_threads>>>(ddist, i, n);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed Time: %f\n", elapsedTime);

    cudaMemcpy(Dist, ddist, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    
    clock_gettime(CLOCK_MONOTONIC, &io_outstart);
    output(argv[2]);
    clock_gettime(CLOCK_MONOTONIC, &io_outend);
    io_elapsed += (io_outend.tv_sec - io_outstart.tv_sec) + (io_outend.tv_nsec - io_outstart.tv_nsec) / 1e9;
    printf("IO Elapsed Time: %f\n", io_elapsed);
    return 0;
}