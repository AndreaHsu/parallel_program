#include <bits/stdc++.h>
#include <mpi.h>
#include <boost/sort/spreadsort/float_sort.hpp>
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

void debugPrint(int rank, int n, int s, float* buf, string c){
    cout << "Debug Print MyRank: " << rank  << ", recieved from OtherRank: " << n << ", case: " << c << endl;
    for(int i = 0; i < s; i++){
        cout << buf[i] << " ";
    }
    cout << endl;
}

int main(int argc, char ** argv){
    MPI_Init(&argc, &argv);
    float startCal = MPI_Wtime();
    int rank;
    int numOfProcess;
    int arrSize = atoi(argv[1]);
    int localSize;
    int localOffset;
    float* localDataBuf;
    /*Time*/
    float startTime;
    float endTime;
    float totalCalculateTime;
    float totalCommunicateTime;
    float totalIoTime;
    float localCalculateTime = 0.0;
    float localCommunicateTime = 0.0;
    float localIoTime = 0.0;

    MPI_Comm_size(MPI_COMM_WORLD, &numOfProcess);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if(numOfProcess > arrSize) numOfProcess = arrSize;
    // 1. Partition data
    int quotient = arrSize / numOfProcess;
    int remainder = arrSize % numOfProcess; 
    localSize = quotient + (rank < remainder);
    localOffset = (rank < remainder) ? (sizeof(float) * rank * localSize) : (sizeof(float) * (remainder + rank * localSize));
    localDataBuf = new float[localSize];

    if(rank >= arrSize) {
        localOffset = 0;
        localSize = 0;
        rank = -1;
    }
    
    // 2. Read data
    MPI_File fin;
    startTime = MPI_Wtime();
    MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);
    MPI_File_read_at(fin, localOffset, localDataBuf, localSize, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fin);
    endTime = MPI_Wtime();
    localIoTime += endTime - startTime;
    
    // if(localSize) radix_sort(localDataBuf, localSize);
    if(localSize) boost::sort::spreadsort::float_sort(localDataBuf, localDataBuf+localSize);

    // 3. Send and Recv data between neighbor node: find neighbor and the localSize of neighbor
    int leftNeighbor = rank - 1; 
    int rightNeighbor = (rank == (numOfProcess - 1)) ? MPI_PROC_NULL : (rank + 1);
    int leftSize = quotient + (leftNeighbor < remainder);;
    int rightSize = quotient + (rightNeighbor < remainder);;
    float* recvDataBuf = new float[leftSize];
    
    // 4. Odd Even Sort
    float* mergeDataBuf = new float[localSize];
    int maxSortNum = numOfProcess+1;
    int stage = rank & 1;
    int idxA, idxB, idxC;
    MPI_Request sendReq1, recvReq1, sendReq2, recvReq2;

    while(maxSortNum--){
        if(rank == -1) break;
        if(stage && rightNeighbor != -1){
            // get the smaller half
            // take the head and buttom to decide how many data need to exchange
            startTime = MPI_Wtime();
            MPI_Sendrecv(&localDataBuf[localSize-1], 1, MPI_FLOAT, rightNeighbor, COMM_STAGE, recvDataBuf, 1, MPI_FLOAT, rightNeighbor, COMM_STAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            endTime = MPI_Wtime();
            localCommunicateTime += endTime - startTime;
            
            if(recvDataBuf[0] < localDataBuf[localSize-1]){
                startTime = MPI_Wtime();
                MPI_Isend(localDataBuf, localSize, MPI_FLOAT, rightNeighbor, COMM_STAGE, MPI_COMM_WORLD, &sendReq1);
                MPI_Irecv(recvDataBuf, rightSize, MPI_FLOAT, rightNeighbor, COMM_STAGE, MPI_COMM_WORLD, &recvReq1);
                MPI_Wait(&sendReq1, MPI_STATUS_IGNORE);
                MPI_Wait(&recvReq1, MPI_STATUS_IGNORE);
                endTime = MPI_Wtime();
                localCommunicateTime += endTime - startTime;
            
                // MPI_Sendrecv(localDataBuf, localSize-1, MPI_FLOAT, rightNeighbor, COMM_STAGE, &recvDataBuf[1], rightSize-1, MPI_FLOAT, rightNeighbor, COMM_STAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                idxA = 0;
                idxB = 0;
                idxC = 0;
                while(idxC != localSize){
                    if(localDataBuf[idxA] < recvDataBuf[idxB]){
                        mergeDataBuf[idxC++] = localDataBuf[idxA++];
                        if(idxA == localSize) break;
                    }else{
                        mergeDataBuf[idxC++] = recvDataBuf[idxB++];
                        if(idxB == rightSize) break;
                    }
                }

                while(idxC != localSize){
                    mergeDataBuf[idxC++] = localDataBuf[idxA++];
                }

                swap(localDataBuf, mergeDataBuf);
            }
        }
        if(!stage && leftNeighbor != -1){
            // get the larger half
            // take the head and buttom to decide how many data need to exchange
            startTime = MPI_Wtime();
            MPI_Sendrecv(&localDataBuf[0], 1, MPI_FLOAT, leftNeighbor, COMM_STAGE, &recvDataBuf[leftSize-1], 1, MPI_FLOAT, leftNeighbor, COMM_STAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            endTime = MPI_Wtime();
            localCommunicateTime += endTime - startTime;
            if(recvDataBuf[leftSize-1] > localDataBuf[0]){
                startTime = MPI_Wtime();
                MPI_Isend(localDataBuf, localSize, MPI_FLOAT, leftNeighbor, COMM_STAGE, MPI_COMM_WORLD, &sendReq2);
                MPI_Irecv(recvDataBuf, leftSize, MPI_FLOAT, leftNeighbor, COMM_STAGE, MPI_COMM_WORLD, &recvReq2);
                MPI_Wait(&sendReq2, MPI_STATUS_IGNORE);
                MPI_Wait(&recvReq2, MPI_STATUS_IGNORE);
                endTime = MPI_Wtime();
                localCommunicateTime += endTime - startTime;
                // MPI_Sendrecv(&localDataBuf[1], localSize-1, MPI_FLOAT, leftNeighbor, COMM_STAGE, recvDataBuf, leftSize-1, MPI_FLOAT, leftNeighbor, COMM_STAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);    
                idxA = localSize - 1;
                idxB = leftSize - 1;
                idxC = localSize - 1;

                while(idxC != -1){
                    if(localDataBuf[idxA] > recvDataBuf[idxB]){
                        mergeDataBuf[idxC--] = localDataBuf[idxA--];
                        if(idxA == -1) break;
                    }else{
                        mergeDataBuf[idxC--] = recvDataBuf[idxB--];
                        if(idxB == -1) break;
                    }
                }

                while(idxC != -1){
                    mergeDataBuf[idxC--] = localDataBuf[idxA--];
                }
                swap(localDataBuf, mergeDataBuf);
            }
        }
        stage ^= 1;
    }

    // 5. Write data
    MPI_File fout;
    startTime = MPI_Wtime();
    MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fout);
    MPI_File_write_at(fout, localOffset, localDataBuf, localSize, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fout);
    endTime = MPI_Wtime();
    localIoTime += endTime - startTime;
    float endCal = MPI_Wtime();
    localCalculateTime += endCal - startCal;
    MPI_Reduce(&localCalculateTime, &totalCalculateTime, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localCommunicateTime, &totalCommunicateTime, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localIoTime, &totalIoTime, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        float tmpTotal = totalCalculateTime/numOfProcess;
        float tmpCom = totalCommunicateTime/numOfProcess;
        float tmpIO  = totalIoTime/numOfProcess;
        cout << tmpTotal << ","
             << tmpTotal - tmpCom - tmpIO << ","
             << tmpCom << ","
             << tmpIO << endl;
        // cout << "Total calculate time: " << totalCalculateTime/numOfProcess - tmpCom - tmpIO<< endl
        //      << "Total communicate time: " << tmpCom << endl
        //      << "Total io time: " << tmpIO << endl;
    }

    MPI_Finalize();
    return 0;
}
