#include <cstdio>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <vector>

#define CHECK_CUDA(func)  {                                                \
  cudaError_t status = (func);                                             \
  if (status != cudaSuccess) {                                             \
    printf("CUDA API failed at line %d with error: %s\n",                  \
           __LINE__, cudaGetErrorString(status));                          \
    return EXIT_FAILURE;                                                   \
  }                                                                         \
}

#define CHECK_CUSPARSE(func)  {                                             \
  cusparseStatus_t status = (func);                                         \
  if (status != CUSPARSE_STATUS_SUCCESS) {                                  \
    printf("CUSPARSE API failed at line %d with error.\n",                  \
           __LINE__);                                                       \
    return EXIT_FAILURE;                                                    \
  }                                                                         \
}

int main(void) {
  // Define your problem and inputs (matrix and vectors) here.

  // Host problem definition
  float alpha = 1.0f;
  float beta = 0.0f;

  // Matrix definition
  const int num_matrices = 3;
  int num_rows[num_matrices] = {4, 4, 4};
  int num_cols[num_matrices] = {4, 4, 4};
  int nnz[num_matrices] = {9, 9, 9};
  // Same sparse structure for all three matrices for simplicity
  int h_rows[num_matrices][9] = {{0, 0, 0, 1, 2, 2, 2, 3, 3},
                                 {0, 0, 0, 1, 2, 2, 2, 3, 3},
                                 {0, 0, 0, 1, 2, 2, 2, 3, 3}};
  int h_columns[num_matrices][9] = {{0, 2, 3, 1, 0, 2, 3, 1, 3},
                                    {0, 2, 3, 1, 0, 2, 3, 1, 3},
                                    {0, 2, 3, 1, 0, 2, 3, 1, 3}};
  float h_values[num_matrices][9] = {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                      6.0f, 7.0f, 8.0f, 9.0f},
                                     {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                      6.0f, 7.0f, 8.0f, 9.0f},
                                     {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                      6.0f, 7.0f, 8.0f, 9.0f}};
  float hX[num_matrices][4] = {{1.0f, 2.0f, 3.0f, 4.0f},
                               {1.0f, 2.0f, 3.0f, 4.0f},
                               {1.0f, 2.0f, 3.0f, 4.0f}};
  float hY[num_matrices][4] = {{0.0f, 0.0f, 0.0f, 0.0f},
                               {0.0f, 0.0f, 0.0f, 0.0f},
                               {0.0f, 0.0f, 0.0f, 0.0f}};

  float hY_result[num_matrices][4] = {{19.0f, 8.0f, 51.0f, 52.0f },
                               {19.0f, 8.0f, 51.0f, 52.0f },
                               {19.0f, 8.0f, 51.0f, 52.0f }};


// Define device arrays/vectors to hold matrices and vector data
std::vector<int*> dA_rows(num_matrices), dA_columns(num_matrices);
std::vector<float*> dA_values(num_matrices), dX(num_matrices), dY(num_matrices);

// Create arrays of streams and cusparseHandles
std::vector<cudaStream_t> streams(num_matrices);
std::vector<cusparseHandle_t> handles(num_matrices);
std::vector<cusparseSpMatDescr_t> matA(num_matrices);
std::vector<cusparseDnVecDescr_t> vecX(num_matrices);
std::vector<cusparseDnVecDescr_t> vecY(num_matrices);

std::vector<void*> dBuffers(num_matrices);
std::vector<size_t> bufferSize(num_matrices);
for (int i = 0; i < num_matrices; i++) {

    cudaStreamCreate(&streams[i]);
    cusparseCreate(&handles[i]);
    cusparseSetStream(handles[i], streams[i]);

    // Allocate device memory for each matrix and vector
    CHECK_CUDA(cudaMalloc((void**) &dA_rows[i], nnz[i] * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void**) &dA_columns[i], nnz[i] * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void**) &dA_values[i], nnz[i] * sizeof(float)))
    CHECK_CUDA(cudaMalloc((void**) &dX[i], num_cols[i] * sizeof(float)))
    CHECK_CUDA(cudaMalloc((void**) &dY[i], num_rows[i] * sizeof(float)))

    // Copy matrices and vectors from host to device
    // I think this can be done asynchronously Keep it for now
    CHECK_CUDA(cudaMemcpy(dA_rows[i], h_rows[i], nnz[i] * sizeof(int), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dA_columns[i], h_columns[i], nnz[i] * sizeof(int), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dA_values[i], h_values[i], nnz[i] * sizeof(float), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dX[i], hX[i], num_cols[i] * sizeof(float), cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dY[i], hY[i], num_rows[i] * sizeof(float), cudaMemcpyHostToDevice))

    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCoo(&matA[i], num_rows[i], num_cols[i], nnz[i],
                                      dA_rows[i], dA_columns[i], dA_values[i],
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

    // Create dense vector X
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX[i], num_cols[i], dX[i], CUDA_R_32F) )

    // Create dense vector Y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY[i], num_rows[i], dY[i], CUDA_R_32F) )

    // Allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                    handles[i], CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA[i], vecX[i], &beta, vecY[i], CUDA_R_32F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize[i]) )

    CHECK_CUDA( cudaMalloc(&dBuffers[i], bufferSize[i]) )
}



  for (int i = 0; i < num_matrices; i++) {

    CHECK_CUSPARSE( cusparseSpMV(
                                    handles[i], CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA[i], vecX[i], &beta, vecY[i], CUDA_R_32F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA[i]) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX[i]) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY[i]) )

  }

  // Copy the results back from the device
  for (int i = 0; i < num_matrices; i++) {
    CHECK_CUDA(cudaMemcpy(hY[i], dY[i], num_rows[i] * sizeof(float), cudaMemcpyDeviceToHost))
  }

  // Synchronize each stream
  for (auto &stream : streams) {
    cudaStreamSynchronize(stream);
  }

  for (int i = 0; i < num_matrices; i++) {

    int correct = 1;
    for (int j = 0; j < num_rows[i]; j++) {
        if (hY[i][j] != hY_result[i][j]) { // direct floating point comparison is not
            correct = 0;             // reliable
            break;
        }
    }
    if (correct)
        printf("spmv_coo_example test PASSED\n");
    else
        printf("spmv_coo_example test FAILED: wrong result\n");
  }

  // Destroy handles and streams
  for(auto &handle : handles) {
    cusparseDestroy(handle);
  }

  for(auto &stream : streams) {
    cudaStreamDestroy(stream);
  }

  // Copy the results back from the device
  // Place your codes here

  return 0;
}
