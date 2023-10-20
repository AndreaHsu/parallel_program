#include <bits/stdc++.h>
#include <mpi.h>
#include <boost/sort/spreadsort/spreadsort.hpp>
using namespace std;
#define COMM_STAGE 1

void radix_sort(float* arr, int len){
    for(int i = 0; i < len; i++){
        reinterpret_cast<int &>(arr[i]) = (reinterpret_cast<int &>(arr[i]) >> 31 & 0x1) ? (~reinterpret_cast<int &>(arr[i])) : (reinterpret_cast<int &>(arr[i]) | 0x80000000);
    }
    vector<float> bucket[256];
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < len; j++){
            bucket[(reinterpret_cast<int &>(arr[j]) >> (i * 8)) & 0xff].push_back(arr[j]);
        }
        int idx = 0;
        int bucketLen = 0;
        for(int j = 0; j < 256; j++){
            bucketLen = bucket[j].size();
            for(int k = 0; k < bucketLen; k++){
                arr[idx++] = bucket[j][k];
            }
            bucket[j].clear();
        }
    }
    for(int i = 0; i < len; i++){
        reinterpret_cast<int &>(arr[i]) = (reinterpret_cast<int &>(arr[i]) >> 31 & 0x1) ? (reinterpret_cast<int &>(arr[i]) & 0x7fffffff) : (~reinterpret_cast<int &>(arr[i]));
    }
}

void merge_large(int sizeA, int sizeB, int sizeC, float* dataA, float* dataB, float* dataC){
    int idxA = sizeA - 1;
    int idxB = sizeB - 1;
    int idxC = sizeC - 1;

    while(idxA >= 0 && idxB >= 0){// && idxC >= 0){
        if(dataA[idxA] > dataB[idxB]){
            dataC[idxC--] = dataA[idxA--];
        }else{
            dataC[idxC--] = dataB[idxB--];
        }
        if(idxC < 0) return;
    }

    while(idxC >= 0){
        dataC[idxC--] = dataA[idxA--];
    }

    return ;
}

void merge_small(int sizeA, int sizeB, int sizeC, float* dataA, float* dataB, float* dataC){
    int idxA = 0;
    int idxB = 0;
    int idxC = 0;

    while(idxA < sizeA && idxB < sizeB){// && idxC < sizeC){
        if(dataA[idxA] < dataB[idxB]){
            dataC[idxC++] = dataA[idxA++];
        }else{
            dataC[idxC++] = dataB[idxB++];
        }
        if(idxC >= sizeC) return;
    }

    while(idxC < sizeC){
        dataC[idxC++] = dataA[idxA++];
    }

   return;
}

void debugPrint(int rank, int n, int s, float* buf, string c){
    cout << "Debug Print MyRank: " << rank  << ", recieved from OtherRank: " << n << ", case: " << c << endl;
    for(int i = 0; i < s; i++){
        cout << buf[i] << " ";
    }
    cout << endl;
}

int main(int argc, char ** argv){
    MPI_Init(&argc, &argv);
    int rank;
    int numpOfProcess;
    int arrSize = atoi(argv[1]);
    int localSize;
    int localOffset;
    float* localDataBuf;

    MPI_Comm_size(MPI_COMM_WORLD, &numpOfProcess);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if(numpOfProcess > arrSize) numpOfProcess = arrSize;
    // 1. Partition data
    int quotient = arrSize / numpOfProcess;
    int remainder = arrSize % numpOfProcess; 
    localSize = quotient + (rank < remainder);
    localOffset = (rank < remainder) ? (sizeof(float) * rank * localSize) : (sizeof(float) * (remainder + rank * localSize));
    localDataBuf = new float[localSize];

    if(rank >= arrSize) {
        localSize = 0;
        rank = -1;
    }
    
    // 2. Read data
    MPI_File fin;
    MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);
    MPI_File_read_at(fin, localOffset, localDataBuf, localSize, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fin);
    
    // if(localSize) radix_sort(localDataBuf, localSize);
    if(localSize) boost::sort::spreadsort::spreadsort(localDataBuf, localDataBuf+localSize);

    // 3. Send and Recv data between neighbor node: find neighbor and the localSize of neighbor
    int leftNeighbor = rank - 1; 
    int rightNeighbor = (rank == (numpOfProcess - 1)) ? MPI_PROC_NULL : (rank + 1);
    int leftSize = localSize + (rank == remainder);
    int rightSize = localSize - (rank+1 == remainder);
    float* recvDataBuf = new float[leftSize];
    
    // 4. Odd Even Sort
    float* mergeDataBuf = new float[localSize];
    int maxSortNum = numpOfProcess+1;
    int stage = rank & 1;
    while(maxSortNum--){
        if(rank == -1) break;
        if(stage && rightNeighbor != -1){
            // get the smaller half
            // take the head and buttom to decide how many data need to exchange
            MPI_Sendrecv(&localDataBuf[localSize-1], 1, MPI_FLOAT, rightNeighbor, COMM_STAGE, recvDataBuf, 1, MPI_FLOAT, rightNeighbor, COMM_STAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(recvDataBuf[0] <= localDataBuf[localSize-1]){
                MPI_Sendrecv(localDataBuf, localSize, MPI_FLOAT, rightNeighbor, COMM_STAGE, recvDataBuf, rightSize, MPI_FLOAT, rightNeighbor, COMM_STAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                merge_small(localSize, rightSize, localSize, localDataBuf, recvDataBuf, mergeDataBuf);
                swap(localDataBuf, mergeDataBuf);
            }
        }
        if(!stage && leftNeighbor != -1){
            // get the larger half
            // take the head and buttom to decide how many data need to exchange
            MPI_Sendrecv(&localDataBuf[0], 1, MPI_FLOAT, leftNeighbor, COMM_STAGE, recvDataBuf, 1, MPI_FLOAT, leftNeighbor, COMM_STAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(recvDataBuf[0] >= localDataBuf[0]){
                MPI_Sendrecv(localDataBuf, localSize, MPI_FLOAT, leftNeighbor, COMM_STAGE, recvDataBuf, leftSize, MPI_FLOAT, leftNeighbor, COMM_STAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);    
                merge_large(localSize, leftSize, localSize, localDataBuf, recvDataBuf, mergeDataBuf);
                swap(localDataBuf, mergeDataBuf);
            }
        }
        stage ^= 1;
    }

    // 5. Write data
    MPI_File fout;
    MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fout);
    MPI_File_write_at(fout, localOffset, localDataBuf, localSize, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fout);
    MPI_Finalize();
    return 0;
}
