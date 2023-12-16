#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <math.h>
#include <iostream>
#include <omp.h>
#define INF ((1 << 30) - 1)

using namespace std;
int numOfThread, v, e;
int dis[6001][6001];

void read_input(const char* filename){
    FILE* infile;
    infile = fopen(filename, "rb");
    fread(&v, sizeof(int), 1, infile);
    fread(&e, sizeof(int), 1, infile);
    for(int i = 0; i < v; i++){
        for(int j = 0; j < v; j++){
            if(i == j) dis[i][j] = 0;
            else dis[i][j] = INF;
        }
    }
    int edge[3];
    for(int i = 0; i < e; i++){
        fread(edge, sizeof(int), 3, infile);
        dis[edge[0]][edge[1]] = edge[2];
    }
    fclose(infile);
    return;
}

void write_output(const char* filename){
    FILE* outfile = fopen(filename, "wb");
    for(int i = 0; i < v; i++){
        fwrite(dis[i], sizeof(int), v, outfile);
    }
    fclose(outfile);
}

int main(int argc, char** argv){
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    numOfThread = CPU_COUNT(&cpu_set);
    read_input(argv[1]);
    
    for(int k = 0; k < v; k++){
        #pragma omp parallel for schedule(static) num_threads(numOfThread)
            for(int i = 0; i < v; i++){
                for(int j = 0; j < v; j++){
                    dis[i][j] = min(dis[i][j], dis[i][k]+dis[k][j]);
                }
            }
    }

    write_output(argv[2]);
    return 0;
}