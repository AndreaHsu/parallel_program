#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

//======================
#define DEV_NO 0
cudaDeviceProp prop;
const int INF = ((1 << 30) - 1);
const int Blocksize = 32;
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
    int x = threadIdx.x;
    int y = threadIdx.y;

    int realx = threadIdx.x + round * Blocksize;
    int realy = threadIdx.y + round * Blocksize;

    shr[y][x] = dist[realx * n + realy];

    __syncthreads();

    for(int i = 0; i < Blocksize; i++){
        shr[y][x] = min(shr[y][x], shr[y][i] + shr[i][x]);
        __syncthreads();
    }
    
    dist[realx * n + realy] = shr[y][x];
    return;
}

__global__ void block_FW_p2(int* dist, int round, int n){
    if(blockIdx.y == round) return;
    __shared__ int shr[Blocksize][Blocksize];
    __shared__ int row[Blocksize][Blocksize];
    __shared__ int col[Blocksize][Blocksize];
    int x = threadIdx.x;
    int y = threadIdx.y;

    int realx = round * Blocksize + threadIdx.x;
    int realy = round * Blocksize + threadIdx.y;
    int realSameRowPos = realx * n + (blockIdx.y * Blocksize + threadIdx.y);
    int realSameColPos = (blockIdx.y * Blocksize + threadIdx.x) * n + realy;
    
    shr[y][x] = dist[realx * n + realy];
    row[y][x] = dist[realSameRowPos]; 
    col[y][x] = dist[realSameColPos];

    __syncthreads();

    #pragma unroll 32
    for(int i = 0; i < Blocksize; i++){
        row[y][x] = min(row[y][x], row[y][i] + shr[i][x]);
        col[y][x] = min(col[y][x], shr[y][i] + col[i][x]);
    }
    
    dist[realSameRowPos] = row[y][x];
    dist[realSameColPos] = col[y][x];
    return;
}

__global__ void block_FW_p3(int* dist, int round, int n){
    if(blockIdx.x == round || blockIdx.y == round) return;
    __shared__ int shr[Blocksize][Blocksize];
    __shared__ int row[Blocksize][Blocksize];
    __shared__ int col[Blocksize][Blocksize];
    int x = threadIdx.x;
    int y = threadIdx.y;

    int realx = blockIdx.x * Blocksize + threadIdx.x;
    int realy = blockIdx.y * Blocksize + threadIdx.y;
    // if(realx >= n || realy >= n) return;
    int realSameRowPos = (round * Blocksize + threadIdx.x) * n + realy;
    int realSameColPos = realx * n + (round * Blocksize + threadIdx.y);
    
    shr[y][x] = dist[realx * n + realy];
    row[y][x] = dist[realSameRowPos]; 
    col[y][x] = dist[realSameColPos];

    __syncthreads();

    #pragma unroll 32
    for(int i = 0; i < Blocksize; i++){
        shr[y][x] = min(shr[y][x], row[y][i] + col[i][x]);
    }
    
    dist[realx * n + realy] = shr[y][x];
    return;
}


int main(int argc, char* argv[]) {
    input(argv[1]);
    int* ddist;
    cudaMalloc(&ddist, n * n * sizeof(int));
    cudaMemcpy(ddist, Dist, n * n * sizeof(int), cudaMemcpyHostToDevice);

    // cudaGetDeviceProperties(&prop, DEV_NO);
    // printf("maxThreasPerBlock = %d, sharedMemPerBlock = %d", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);
    // maxThreasPerBlock = 1024, sharedMemPerBlock = 49152
    int B = n / Blocksize;
    dim3 num_blocks_p1(1, 1);
    dim3 num_blocks_p2(1, B);
    dim3 num_blocks_p3(B, B);
    dim3 num_threads(32, 32);

    for(int i = 0; i < B; i++){
        block_FW_p1<<<num_blocks_p1, num_threads>>>(ddist, i, n);
        block_FW_p2<<<num_blocks_p2, num_threads>>>(ddist, i, n);
        block_FW_p3<<<num_blocks_p3, num_threads>>>(ddist, i, n);
    }

    cudaMemcpy(Dist, ddist, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    output(argv[2]);
    return 0;
}