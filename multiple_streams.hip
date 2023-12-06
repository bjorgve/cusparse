
#include <hip/hip_runtime.h>
#include <cstdio>
#include <hip/hip_runtime_api.h>
#include <hipsparse.h>
#include <vector>
#include <array>
#include <iostream>

#define CHECK_HIP(func, error_handler) {                                   \
    hipError_t status = (func);                                            \
    if (status != hipSuccess) {                                            \
        printf("HIP API failed at line %d with error: %s\n", __LINE__,     \
               hipGetErrorString(status));                                 \
        error_handler(status);                                              \
    }                                                                       \
}

#define CHECK_CUSPARSE(func, error_handler) {                               \
    hipsparseStatus_t status = (func);                                       \
    if (status != HIPSPARSE_STATUS_SUCCESS) {                                \
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
    std::vector<hipStream_t> streams(num_matrices);
    std::vector<hipsparseHandle_t> handles(num_matrices);
    std::vector<hipsparseSpMatDescr_t> matA(num_matrices);
    std::vector<hipsparseDnVecDescr_t> vecX(num_matrices);
    std::vector<hipsparseDnVecDescr_t> vecY(num_matrices);

    std::vector<void*> dBuffers(num_matrices);
    std::vector<size_t> bufferSize(num_matrices);

    // // Allocate device memory for each matrix and vector
    // for (auto i = 0; i < num_matrices; i++) {
    //     CHECK_HIP(hipMalloc((void**) &dA_rows[i], data.nnz[i] * sizeof(int)), error_handler)
    //     CHECK_HIP(hipMalloc((void**) &dA_columns[i], data.nnz[i] * sizeof(int)), error_handler)
    //     CHECK_HIP(hipMalloc((void**) &dA_values[i], data.nnz[i] * sizeof(float)), error_handler)
    //     CHECK_HIP(hipMalloc((void**) &dX[i], data.num_cols[i] * sizeof(float)), error_handler)
    //     CHECK_HIP(hipMalloc((void**) &dY[i], data.num_rows[i] * sizeof(float)), error_handler)
    //     CHECK_HIP(hipMalloc(&dBuffers[i], bufferSize[i]), error_handler )
    // }

    for (int i = 0; i < num_matrices; i++) {

        // hipStreamCreate(&streams[i]);
        hipError_t err;
        hipsparseStatus_t sp_status;


        err = hipStreamCreate(&streams[i]);
        if (err != hipSuccess) {
            printf("hipStreamCreate failed with %s\n", hipGetErrorString(err));
            exit(1);
        }

        sp_status  = hipsparseCreate(&handles[i]);
        if (sp_status != HIPSPARSE_STATUS_SUCCESS) {
            printf("hipsparseCreate failed with %s\n", hipGetErrorString(err));
            exit(1);
        }


        sp_status = hipsparseSetStream(handles[i], streams[i]);
        if (sp_status != HIPSPARSE_STATUS_SUCCESS) {
            printf("hipsparseSetStream failed with %s\n", hipGetErrorString(err));
            exit(1);
        }

            std::cout << "Allocating memory for buffer " << i
              << ": size = " << bufferSize[i]
              << ", stream = " << streams[i]
              << std::endl;

        // Allocate device memory for each matrix and vector
        CHECK_HIP(hipMallocAsync((void**) &dA_rows[i], data.nnz[i] * sizeof(int), streams[i]), error_handler)
        CHECK_HIP(hipMallocAsync((void**) &dA_columns[i], data.nnz[i] * sizeof(int), streams[i]), error_handler)
        CHECK_HIP(hipMallocAsync((void**) &dA_values[i], data.nnz[i] * sizeof(float), streams[i]), error_handler)
        CHECK_HIP(hipMallocAsync((void**) &dX[i], data.num_cols[i] * sizeof(float), streams[i]), error_handler)
        CHECK_HIP(hipMallocAsync((void**) &dY[i], data.num_rows[i] * sizeof(float), streams[i]), error_handler)
        // CHECK_HIP(hipMallocAsync(&dBuffers[i], bufferSize[i], streams[i]), error_handler )

        // Copy matrices and vectors from host to device asynchronously
        CHECK_HIP(hipMemcpyAsync(dA_rows[i], data.h_rows[i].data(), data.nnz[i] * sizeof(int), hipMemcpyHostToDevice, streams[i]), error_handler)
        CHECK_HIP(hipMemcpyAsync(dA_columns[i], data.h_columns[i].data(), data.nnz[i] * sizeof(int), hipMemcpyHostToDevice, streams[i]), error_handler)
        CHECK_HIP(hipMemcpyAsync(dA_values[i], data.h_values[i].data(), data.nnz[i] * sizeof(float), hipMemcpyHostToDevice, streams[i]), error_handler)
        CHECK_HIP(hipMemcpyAsync(dX[i], data.hX[i].data(), data.num_cols[i] * sizeof(float), hipMemcpyHostToDevice, streams[i]), error_handler)
        CHECK_HIP(hipMemcpyAsync(dY[i], data.hY[i].data(), data.num_rows[i] * sizeof(float), hipMemcpyHostToDevice, streams[i]), error_handler)

        // Create sparse matrix A in CSR format
        CHECK_CUSPARSE( hipsparseCreateCoo(&matA[i], data.num_rows[i], data.num_cols[i], data.nnz[i],
                                          dA_rows[i], dA_columns[i], dA_values[i],
                                          HIPSPARSE_INDEX_32I,
                                          HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F), error_handler)

        // Create dense vector X
        CHECK_CUSPARSE( hipsparseCreateDnVec(&vecX[i], data.num_cols[i], dX[i], HIP_R_32F), error_handler )

        // Create dense vector Y
        CHECK_CUSPARSE( hipsparseCreateDnVec(&vecY[i], data.num_rows[i], dY[i], HIP_R_32F), error_handler )

        // Allocate an external buffer if needed
        CHECK_CUSPARSE( hipsparseSpMV_bufferSize(
                                        handles[i], HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, matA[i], vecX[i], &beta, vecY[i], HIP_R_32F,
                                        HIPSPARSE_SPMV_ALG_DEFAULT, &bufferSize[i]), error_handler )
        CHECK_HIP(hipMallocAsync(&dBuffers[i], bufferSize[i], streams[i]), error_handler);


        CHECK_CUSPARSE( hipsparseSpMV(
                                        handles[i], HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, matA[i], vecX[i], &beta, vecY[i], HIP_R_32F,
                                        HIPSPARSE_SPMV_ALG_DEFAULT, &bufferSize), error_handler )

        // destroy matrix/vector descriptors
        CHECK_CUSPARSE( hipsparseDestroySpMat(matA[i]), error_handler)
        CHECK_CUSPARSE( hipsparseDestroyDnVec(vecX[i]), error_handler)
        CHECK_CUSPARSE( hipsparseDestroyDnVec(vecY[i]), error_handler)

        CHECK_HIP(hipMemcpyAsync(data.hY[i].data(), dY[i], data.num_rows[i] * sizeof(float), hipMemcpyDeviceToHost, streams[i]), error_handler)
    }


    // Synchronize each stream
    for (auto &stream : streams) {
        hipStreamSynchronize(stream);
    }



    for(int i = 0; i < num_matrices; i++) {
        CHECK_CUSPARSE(hipsparseDestroy(handles[i]), error_handler);
        CHECK_HIP(hipStreamDestroy(streams[i]), error_handler);
    }
    // Free device memory
    for(auto i=0; i < num_matrices; i++) {
        CHECK_HIP(hipFree(dA_rows[i]), error_handler);
        CHECK_HIP(hipFree(dA_columns[i]), error_handler);
        CHECK_HIP(hipFree(dA_values[i]), error_handler);
        CHECK_HIP(hipFree(dX[i]), error_handler);
        CHECK_HIP(hipFree(dY[i]), error_handler);
        CHECK_HIP(hipFree(dBuffers[i]), error_handler);
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
  CHECK_HIP(hipGetDeviceCount(&num_gpus), error_handler);
  std::cout << "num_gpus " << num_gpus << std::endl;

  std::vector<MatrixData> gpu_data;
  for(int i=0; i < num_gpus; i++) {
      gpu_data.push_back(generate_identical_matrices(num_matrices_per_gpu));
  }

  for(int i=0; i < num_gpus; i++) {
      CHECK_HIP(hipSetDevice(i), error_handler);
      multiply_matrices(gpu_data[i], num_matrices_per_gpu, alpha, beta);
  }

  return 0;
}
