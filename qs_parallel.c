#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

double Myrandom(void) {
    int Sign = rand() % 100;
    return Sign;
}

void swap(int* a, int* b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

int partition(int arr[], int low, int high, int pivot) {
    int pivotValue = arr[pivot];
    swap(&arr[pivot], &arr[high]);  //moving the pivot to the end of the array
    int index = low;

    for (int i = low; i < high; i++) {
        if (arr[i] < pivotValue) {
            swap(&arr[i], &arr[index]);
            index++;
        }
    }

    swap(&arr[index], &arr[high]);  // placing the pivot in the right position
    return index;
}
void quicksort(int *array, int low, int high) {
    int i = low, j = high;
    int pivot = array[(low + high) / 2];
    int temp;
    
    while (i <= j) {
        while (array[i] < pivot)
            i++;
        while (array[j] > pivot)
            j--;
        if (i <= j) {
            temp = array[i];
            array[i] = array[j];
            array[j] = temp;
            i++;
            j--;
        }
    }
    
    if (low < j)
        quicksort(array, low, j);
    if (i < high)
        quicksort(array, i, high);
}

void psrs(int *array, int arraySize, int NT, int rank, int numProcesses) {
    int start_index = rank * (arraySize / numProcesses);
    int end_index = start_index + (arraySize / numProcesses) - 1;
    
    int *pivots = (int*)malloc((NT - 1) * sizeof(int));
    int *pivotSamples = (int*)malloc(NT * NT * sizeof(int));
    
    int **count = (int**)malloc(NT * sizeof(int*));
    for (int i = 0; i < NT; i++) {
        count[i] = (int*)calloc(NT, sizeof(int));
    }
    
    int ***subLists = (int***)malloc(NT * sizeof(int**));
    for (int i = 0; i < NT; i++) {
        subLists[i] = (int**)malloc(NT * sizeof(int*));
        for (int j = 0; j < NT; j++) {
            subLists[i][j] = (int*)malloc(arraySize * sizeof(int));
        }
    }
    
    int num_elements_per_process = arraySize / numProcesses;
    
    quicksort(array, start_index, end_index);
    
    // First Phase
    #pragma omp parallel num_threads(NT) shared(array, subLists, count)
    {
        int tid = omp_get_thread_num();
        int chunk_start_index = start_index + tid * num_elements_per_process / NT;
        int chunk_end_index = start_index + (tid + 1) * num_elements_per_process / NT - 1;
        if (tid == NT - 1) {
            chunk_end_index = end_index;
        }
        quicksort(array, chunk_start_index, chunk_end_index);
    }
    
    // Second Phase
    for (int j = 0; j < NT; j++) {
        pivotSamples[rank * NT + j] = array[start_index + j * num_elements_per_process / (NT * NT)];
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        quicksort(pivotSamples, 0, (NT * NT) - 1);
    }
    
    MPI_Bcast(pivotSamples, NT * NT, MPI_INT, 0, MPI_COMM_WORLD);
    for (int i = 0; i < NT - 1; i++) {
        pivots[i] = pivotSamples[(i + 1) * NT + (NT / 2 - 1)];
    }
    
    for (int i = start_index; i <= end_index; i++) {
        int j = 0;
        while (array[i] > pivots[j] && j < NT - 1) {
            j++;
        }
        count[rank][j]++;
    }
    
    MPI_Allreduce(MPI_IN_PLACE, count[rank], NT, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    for (int i = 0; i < numProcesses; i++) {
        MPI_Bcast(count[i], NT, MPI_INT, i, MPI_COMM_WORLD);
    }
    
    for (int i = start_index; i <= end_index; i++) {
        int j = 0;
        while (array[i] > pivots[j] && j < NT - 1) {
            j++;
        }
        subLists[rank][j][count[rank][j]] = array[i];
        count[rank][j]++;
    }
    
    for (int i = 0; i < NT; i++) {
        for (int j = 0; j < NT; j++) {
            MPI_Gather(subLists[i][j], count[i][j], MPI_INT, array, count[i][j], MPI_INT, j, MPI_COMM_WORLD);
        }
    }
    
    // Third Phase
    quicksort(array, 0, (num_elements_per_process * numProcesses) - 1);
    
    free(pivots);
    free(pivotSamples);
    
    for (int i = 0; i < NT; i++) {
        free(count[i]);
    }
    free(count);
    
    for (int i = 0; i < NT; i++) {
        for (int j = 0; j < NT; j++) {
            free(subLists[i][j]);
        }
        free(subLists[i]);
    }
    free(subLists);
}

int main(int argc, char** argv) {
    int rank, numProcesses;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int NT = atoi(argv[1]);  // number of threads
    int arraySize = atoi(argv[2]);  // size of the array

    int *array;
    if (rank == 0) {
        array = (int*) malloc(arraySize * sizeof(int));
        srand(time(NULL));
        for (int i = 0; i < arraySize; i++) {
            array[i] = Myrandom();
        }
    }

    MPI_Bcast(&arraySize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int num_elements_per_process = arraySize / numProcesses;

    int* subArray = (int*)malloc(num_elements_per_process * sizeof(int));

    MPI_Scatter(array, num_elements_per_process, MPI_INT, subArray, num_elements_per_process, MPI_INT, 0, MPI_COMM_WORLD);

    psrs(subArray, num_elements_per_process, NT, rank, numProcesses);

    int* sortedArray = NULL;
    if (rank == 0) {
        sortedArray = (int*)malloc(arraySize * sizeof(int));
    }

    MPI_Gather(subArray, num_elements_per_process, MPI_INT, sortedArray, num_elements_per_process, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Merge the sorted sub-arrays
        int* mergedArray = (int*)malloc(arraySize * sizeof(int));
        int* indices = (int*)calloc(numProcesses, sizeof(int));

        for (int i = 0; i < arraySize; i++) {
            int minIndex = -1;
            int minValue = 1000000;

            for (int j = 0; j < numProcesses; j++) {
                if (indices[j] < num_elements_per_process && sortedArray[j * num_elements_per_process + indices[j]] < minValue) {
                    minIndex = j;
                    minValue = sortedArray[j * num_elements_per_process + indices[j]];
                }
            }

            mergedArray[i] = minValue;
            indices[minIndex]++;
        }

        // Print the sorted array
        printf("Sorted Array:\n");
        for (int i = 0; i < arraySize; i++) {
            printf("%d ", mergedArray[i]);
        }
        printf("\n");

        free(array);
        free(sortedArray);
        free(mergedArray);
        free(indices);
    }

    free(subArray);

    MPI_Finalize();
    return 0;
}