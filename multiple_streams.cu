#include <cstdio>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <vector>
#include <array>

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

struct MatrixData {
    int num_matrices;
    std::vector<int> num_rows, num_cols, nnz;
    std::vector<std::vector<int>> h_rows, h_columns;
    std::vector<std::vector<float>> h_values, hX, hY, hY_result;
};

MatrixData generate_identical_matrices(int num_matrices) {
    MatrixData data;
    data.num_matrices = num_matrices;
    data.num_rows = std::vector<int>(num_matrices, 4);
    data.num_cols = std::vector<int>(num_matrices, 4);
    data.nnz = std::vector<int>(num_matrices, 9);

    std::vector<int> sample_rows = {0, 0, 0, 1, 2, 2, 2, 3, 3};
    std::vector<int> sample_cols = {0, 2, 3, 1, 0, 2, 3, 1, 3};
    std::vector<float> sample_vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    std::vector<float> sample_vec = {1.0f, 2.0f, 3.0f, 4.0f};

    for (int i = 0; i < num_matrices; i++) {
        data.h_rows.push_back(sample_rows);
        data.h_columns.push_back(sample_cols);
        data.h_values.push_back(sample_vals);
        data.hX.push_back(sample_vec);
        data.hY.push_back(std::vector<float>(4, 0.0f));
        data.hY_result.push_back({19.0f, 8.0f, 51.0f, 52.0f}); // comme before
    }
    return data;
}

int main(void) {
  // Define your problem and inputs (matrix and vectors) here.

  // Host problem definition
  float alpha = 1.0f;
  float beta = 0.0f;

  auto num_matrices = 5; // use whatever number you want
  MatrixData data = generate_identical_matrices(num_matrices);

  auto num_rows = data.num_rows;
  auto num_cols = data.num_cols;
  auto nnz = data.nnz;
  auto h_rows = data.h_rows;
  auto h_columns = data.h_columns;
  auto h_values = data.h_values;
  auto hX = data.hX;
  auto hY = data.hY;
  auto hY_result = data.hY_result;


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

// Allocate device memory for each matrix and vector
for (auto i = 0; i < num_matrices; i++) {
    CHECK_CUDA(cudaMalloc((void**) &dA_rows[i], nnz[i] * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void**) &dA_columns[i], nnz[i] * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void**) &dA_values[i], nnz[i] * sizeof(float)))
    CHECK_CUDA(cudaMalloc((void**) &dX[i], num_cols[i] * sizeof(float)))
    CHECK_CUDA(cudaMalloc((void**) &dY[i], num_rows[i] * sizeof(float)))
    CHECK_CUDA(cudaMalloc(&dBuffers[i], bufferSize[i]) )
}

for (int i = 0; i < num_matrices; i++) {

    cudaStreamCreate(&streams[i]);
    cusparseCreate(&handles[i]);
    cusparseSetStream(handles[i], streams[i]);

// Copy matrices and vectors from host to device asynchronously
    CHECK_CUDA(cudaMemcpyAsync(dA_rows[i], h_rows[i].data(), nnz[i] * sizeof(int), cudaMemcpyHostToDevice, streams[i]))
    CHECK_CUDA(cudaMemcpyAsync(dA_columns[i], h_columns[i].data(), nnz[i] * sizeof(int), cudaMemcpyHostToDevice, streams[i]))
    CHECK_CUDA(cudaMemcpyAsync(dA_values[i], h_values[i].data(), nnz[i] * sizeof(float), cudaMemcpyHostToDevice, streams[i]))
    CHECK_CUDA(cudaMemcpyAsync(dX[i], hX[i].data(), num_cols[i] * sizeof(float), cudaMemcpyHostToDevice, streams[i]))
    CHECK_CUDA(cudaMemcpyAsync(dY[i], hY[i].data(), num_rows[i] * sizeof(float), cudaMemcpyHostToDevice, streams[i]))

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


    CHECK_CUSPARSE( cusparseSpMV(
                                    handles[i], CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA[i], vecX[i], &beta, vecY[i], CUDA_R_32F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA[i]) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX[i]) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY[i]) )

    CHECK_CUDA(cudaMemcpyAsync(hY[i].data(), dY[i], num_rows[i] * sizeof(float), cudaMemcpyDeviceToHost, streams[i]))
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

  return 0;
}
