#include <bits/stdc++.h>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <mpi.h>
using namespace std;
#define EVEN_STAGE 1
#define ODD_STAGE 2
#define ODD 1
#define EVEN 0

void radix_sort(float arr[], int len){
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
    int flag = (rank % 2) ? ODD : EVEN;
    
    // TODO: ??
    if(numpOfProcess > arrSize) numpOfProcess = arrSize;
    // 1. Partition data
    int quotient = arrSize / numpOfProcess;
    int remainder = arrSize % numpOfProcess; 
    localSize = (rank < remainder) ? (quotient + 1) : (quotient);
    localOffset = (rank < remainder) ? (sizeof(float) * rank * localSize) : (sizeof(float) * (remainder + rank * localSize));
    localDataBuf = (float*) malloc(sizeof(float) * localSize);

    if(rank >= arrSize) localSize = 0;
    // cout << "rank: " << rank << ", localSize: " << localSize << ", localOffset: " << localOffset << endl;
    // 2. Read data
    MPI_File fin;
    MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);
    MPI_File_read_at(fin, localOffset, localDataBuf, localSize, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fin);

    if(localSize) boost::sort::spreadsort::spreadsort(localDataBuf, localDataBuf+localSize);
    // cout << "Before Rank: " << rank << endl;
    // for(int i = 0; i < localSize; i++){
    //     cout << localDataBuf[i] << " ";
    // }
    // cout << endl;

    // 3. Send and Recv data between neighbor node
    // 3-1. find neighbor and the localSize of neighbor
    int evenNeighbor; 
    int oddNeighbor;
    int evenSize;
    int oddSize;
    float* evenRecvBuf;
    float* oddRecvBuf;

    if(flag){
        // rank is odd
        evenNeighbor = rank - 1;
        oddNeighbor = (rank == (numpOfProcess - 1)) ? MPI_PROC_NULL : (rank + 1);
    }else{
        // rank is even
        evenNeighbor = (rank == (numpOfProcess - 1)) ? MPI_PROC_NULL : (rank + 1);
        oddNeighbor = (rank == 0) ? MPI_PROC_NULL : (rank - 1);
    }

    evenSize = (evenNeighbor < remainder) ? (quotient + 1) : (quotient);
    oddSize = (oddNeighbor < remainder) ? (quotient + 1) : (quotient);

    // cout << "Rank: " << rank << ", evenNeighbor: " << evenNeighbor << ", size: " << evenSize << ", oddNeighbor: " << oddNeighbor << ", size: " << oddSize << endl;

    evenRecvBuf = (float*) malloc(sizeof(float) * evenSize);
    oddRecvBuf = (float*) malloc(sizeof(float) * oddSize);
    // 3-2. SendRecv
    
    // 4. Odd Even Sort
    float* mergeDataBuf = (float*) malloc(sizeof(float) * localSize);
    int maxSortNum = numpOfProcess+1;
    MPI_Request sendReq, recvReq;
    
    for(int i = 0; i < maxSortNum; i++){
        if(rank >= numpOfProcess) break;
        if(i % 2){
            // Odd Stage
            if(oddNeighbor < 0) continue;
            MPI_Isend(localDataBuf, localSize, MPI_FLOAT, oddNeighbor, ODD_STAGE, MPI_COMM_WORLD, &sendReq);
            MPI_Irecv(oddRecvBuf, oddSize, MPI_FLOAT, oddNeighbor, ODD_STAGE, MPI_COMM_WORLD, &recvReq);
            MPI_Wait(&sendReq, MPI_STATUS_IGNORE);
            MPI_Wait(&recvReq, MPI_STATUS_IGNORE);
            if(flag){
                // rank is odd, get smaller half
                // debugPrint(rank, oddNeighbor, oddSize, oddRecvBuf, "OddStage-odd-smaller");
                merge_small(localSize, oddSize, localSize, localDataBuf, oddRecvBuf, mergeDataBuf);
            }else{
                // rank is even, get larger half
                // debugPrint(rank, oddNeighbor, oddSize, oddRecvBuf, "OddStage-even-larger");
                merge_large(localSize, oddSize, localSize, localDataBuf, oddRecvBuf, mergeDataBuf);
            }
        }else{
            // Even Stage
            if(evenNeighbor < 0) continue;
            MPI_Isend(localDataBuf, localSize, MPI_FLOAT, evenNeighbor, EVEN_STAGE, MPI_COMM_WORLD, &sendReq);
            MPI_Irecv(evenRecvBuf, evenSize, MPI_FLOAT, evenNeighbor, EVEN_STAGE, MPI_COMM_WORLD, &recvReq);
            MPI_Wait(&sendReq, MPI_STATUS_IGNORE);
            MPI_Wait(&recvReq, MPI_STATUS_IGNORE);
            if(flag){
                // rank is odd, get larger half
                // debugPrint(rank, evenNeighbor, evenSize, evenRecvBuf, "EvenStage-odd-larger");
                merge_large(localSize, evenSize, localSize, localDataBuf, evenRecvBuf, mergeDataBuf);
            }else{
                // rank is even, get smaller half
                // debugPrint(rank, evenNeighbor, evenSize, evenRecvBuf, "EvenStage-even-smaller");
                merge_small(localSize, evenSize, localSize, localDataBuf, evenRecvBuf, mergeDataBuf);
            }
        }
        swap(localDataBuf, mergeDataBuf);
    }
    // 5. Write data
    MPI_File fout;
    MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fout);
    // cout << argv[3] << " , Final rank: " << rank << ", localSize: " << localSize << ", localOffset: " << localOffset << endl;
    MPI_File_write_at(fout, localOffset, localDataBuf, localSize, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fout);
    MPI_Finalize();
    // delete [] localDataBuf;
    // delete [] mergeDataBuf;
    // delete [] evenRecvBuf;
    // delete [] oddRecvBuf;
    return 0;
}
