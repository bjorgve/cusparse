#include <cstdio>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <vector>
#include <array>
#include <iostream>

#define CHECK_CUDA(func, error_handler) {                                   \
    cudaError_t status = (func);                                            \
    if (status != cudaSuccess) {                                            \
        printf("CUDA API failed at line %d with error: %s\n", __LINE__,     \
               cudaGetErrorString(status));                                 \
        error_handler(status);                                              \
    }                                                                       \
}

#define CHECK_CUSPARSE(func, error_handler) {                               \
    cusparseStatus_t status = (func);                                       \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                \
        printf("CUSPARSE API failed at line %d with error %d.\n", __LINE__, \
               status);                                                     \
        error_handler(status);                                              \
    }                                                                       \
}

void error_handler(int status) {
    // Free any resources if needed...

    exit(EXIT_FAILURE);
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

void multiply_matrices(MatrixData& data, int num_matrices, float alpha, float beta) {


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

    // // Allocate device memory for each matrix and vector
    // for (auto i = 0; i < num_matrices; i++) {
    //     CHECK_CUDA(cudaMalloc((void**) &dA_rows[i], data.nnz[i] * sizeof(int)), error_handler)
    //     CHECK_CUDA(cudaMalloc((void**) &dA_columns[i], data.nnz[i] * sizeof(int)), error_handler)
    //     CHECK_CUDA(cudaMalloc((void**) &dA_values[i], data.nnz[i] * sizeof(float)), error_handler)
    //     CHECK_CUDA(cudaMalloc((void**) &dX[i], data.num_cols[i] * sizeof(float)), error_handler)
    //     CHECK_CUDA(cudaMalloc((void**) &dY[i], data.num_rows[i] * sizeof(float)), error_handler)
    //     CHECK_CUDA(cudaMalloc(&dBuffers[i], bufferSize[i]), error_handler )
    // }

    for (int i = 0; i < num_matrices; i++) {

        cudaStreamCreate(&streams[i]);
        cusparseCreate(&handles[i]);
        cusparseSetStream(handles[i], streams[i]);

        // Allocate device memory for each matrix and vector
        CHECK_CUDA(cudaMallocAsync((void**) &dA_rows[i], data.nnz[i] * sizeof(int), streams[i]), error_handler)
        CHECK_CUDA(cudaMallocAsync((void**) &dA_columns[i], data.nnz[i] * sizeof(int), streams[i]), error_handler)
        CHECK_CUDA(cudaMallocAsync((void**) &dA_values[i], data.nnz[i] * sizeof(float), streams[i]), error_handler)
        CHECK_CUDA(cudaMallocAsync((void**) &dX[i], data.num_cols[i] * sizeof(float), streams[i]), error_handler)
        CHECK_CUDA(cudaMallocAsync((void**) &dY[i], data.num_rows[i] * sizeof(float), streams[i]), error_handler)
        CHECK_CUDA(cudaMallocAsync(&dBuffers[i], bufferSize[i], streams[i]), error_handler )

        // Copy matrices and vectors from host to device asynchronously
        CHECK_CUDA(cudaMemcpyAsync(dA_rows[i], data.h_rows[i].data(), data.nnz[i] * sizeof(int), cudaMemcpyHostToDevice, streams[i]), error_handler)
        CHECK_CUDA(cudaMemcpyAsync(dA_columns[i], data.h_columns[i].data(), data.nnz[i] * sizeof(int), cudaMemcpyHostToDevice, streams[i]), error_handler)
        CHECK_CUDA(cudaMemcpyAsync(dA_values[i], data.h_values[i].data(), data.nnz[i] * sizeof(float), cudaMemcpyHostToDevice, streams[i]), error_handler)
        CHECK_CUDA(cudaMemcpyAsync(dX[i], data.hX[i].data(), data.num_cols[i] * sizeof(float), cudaMemcpyHostToDevice, streams[i]), error_handler)
        CHECK_CUDA(cudaMemcpyAsync(dY[i], data.hY[i].data(), data.num_rows[i] * sizeof(float), cudaMemcpyHostToDevice, streams[i]), error_handler)

        // Create sparse matrix A in CSR format
        CHECK_CUSPARSE( cusparseCreateCoo(&matA[i], data.num_rows[i], data.num_cols[i], data.nnz[i],
                                          dA_rows[i], dA_columns[i], dA_values[i],
                                          CUSPARSE_INDEX_32I,
                                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F), error_handler)

        // Create dense vector X
        CHECK_CUSPARSE( cusparseCreateDnVec(&vecX[i], data.num_cols[i], dX[i], CUDA_R_32F), error_handler )

        // Create dense vector Y
        CHECK_CUSPARSE( cusparseCreateDnVec(&vecY[i], data.num_rows[i], dY[i], CUDA_R_32F), error_handler )

        // Allocate an external buffer if needed
        CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                        handles[i], CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, matA[i], vecX[i], &beta, vecY[i], CUDA_R_32F,
                                        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize[i]), error_handler )


        CHECK_CUSPARSE( cusparseSpMV(
                                        handles[i], CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, matA[i], vecX[i], &beta, vecY[i], CUDA_R_32F,
                                        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize), error_handler )

        // destroy matrix/vector descriptors
        CHECK_CUSPARSE( cusparseDestroySpMat(matA[i]), error_handler)
        CHECK_CUSPARSE( cusparseDestroyDnVec(vecX[i]), error_handler)
        CHECK_CUSPARSE( cusparseDestroyDnVec(vecY[i]), error_handler)

        CHECK_CUDA(cudaMemcpyAsync(data.hY[i].data(), dY[i], data.num_rows[i] * sizeof(float), cudaMemcpyDeviceToHost, streams[i]), error_handler)
    }


    // Synchronize each stream
    for (auto &stream : streams) {
        cudaStreamSynchronize(stream);
    }



    for(int i = 0; i < num_matrices; i++) {
        CHECK_CUSPARSE(cusparseDestroy(handles[i]), error_handler);
        CHECK_CUDA(cudaStreamDestroy(streams[i]), error_handler);
    }
    // Free device memory
    for(auto i=0; i < num_matrices; i++) {
        CHECK_CUDA(cudaFree(dA_rows[i]), error_handler);
        CHECK_CUDA(cudaFree(dA_columns[i]), error_handler);
        CHECK_CUDA(cudaFree(dA_values[i]), error_handler);
        CHECK_CUDA(cudaFree(dX[i]), error_handler);
        CHECK_CUDA(cudaFree(dY[i]), error_handler);
        CHECK_CUDA(cudaFree(dBuffers[i]), error_handler);
    }

    for (int i = 0; i < num_matrices; i++) {

        int correct = 1;
        for (int j = 0; j < data.num_rows[i]; j++) {
            if (data.hY[i][j] != data.hY_result[i][j]) { // direct floating point comparison is not
                correct = 0;             // reliable
                break;
            }
        }
        if (correct)
            printf("spmv_coo_example test PASSED\n");
        else
            printf("spmv_coo_example test FAILED: wrong result\n");
    }
}
int main(void) {
  // Host problem definition
  float alpha = 1.0f;
  float beta = 0.0f;
  auto num_matrices_per_gpu = 5; // use assingment per GPU

  int num_gpus = 0;
  CHECK_CUDA(cudaGetDeviceCount(&num_gpus), error_handler);
  std::cout << "num_gpus " << num_gpus << std::endl;

  std::vector<MatrixData> gpu_data;
  for(int i=0; i < num_gpus; i++) {
      gpu_data.push_back(generate_identical_matrices(num_matrices_per_gpu));
  }

  for(int i=0; i < num_gpus; i++) {
      CHECK_CUDA(cudaSetDevice(i), error_handler);
      multiply_matrices(gpu_data[i], num_matrices_per_gpu, alpha, beta);
  }

  return 0;
}
