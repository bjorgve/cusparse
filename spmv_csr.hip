#include <hip/hip_runtime.h>
#include <cstdio>
#include <hip/hip_runtime_api.h>
#include <hipsparse.h>
#include <vector>
#include <array>
#include <iostream>




int main(void) {

      // Host problem definition
    const int A_num_rows      = 4;
    const int A_num_cols      = 4;
    const int A_nnz           = 9;
    int       hA_csrOffsets[] = { 0, 3, 4, 7, 9 };
    int       hA_columns[]    = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
    float     hA_values[]     = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                  6.0f, 7.0f, 8.0f, 9.0f };
    float     hX[]            = { 1.0f, 2.0f, 3.0f, 4.0f };
    float     hY[]            = { 0.0f, 0.0f, 0.0f, 0.0f };
    float     hY_result[]     = { 19.0f, 8.0f, 51.0f, 52.0f };
    float     alpha           = 1.0f;
    float     beta            = 0.0f;


    // Device memory management
    int   *dA_csrOffsets, *dA_columns;
    float *dA_values, *dX, *dY;
    hipMalloc((void**) &dA_csrOffsets,
                (A_num_rows + 1) * sizeof(int));
    hipMalloc((void**) &dA_columns, A_nnz * sizeof(int));
    hipMalloc((void**) &dA_values,  A_nnz * sizeof(float));
    hipMalloc((void**) &dX,         A_num_cols * sizeof(float));
    hipMalloc((void**) &dY,         A_num_rows * sizeof(float));


    // Transfer data from host to device
    hipMemcpy(dA_csrOffsets, hA_csrOffsets,
                (A_num_rows + 1) * sizeof(int),
                hipMemcpyHostToDevice);
    hipMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                hipMemcpyHostToDevice);
    hipMemcpy(dA_values, hA_values, A_nnz * sizeof(float),
                hipMemcpyHostToDevice);
    hipMemcpy(dX, hX, A_num_cols * sizeof(float),
                hipMemcpyHostToDevice);
    hipMemcpy(dY, hY, A_num_rows * sizeof(float),
                           hipMemcpyHostToDevice);

    // HIPSPARSE APIs
    hipsparseHandle_t     handle = NULL;
    hipsparseSpMatDescr_t matA;
    hipsparseDnVecDescr_t vecX, vecY;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;



    hipsparseCreate(&handle);
    hipsparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                        dA_csrOffsets, dA_columns, dA_values,
                        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F);

    // Create dense vector X
    hipsparseCreateDnVec(&vecX, A_num_cols, dX, HIP_R_32F);
    // Create dense vector y
    hipsparseCreateDnVec(&vecY, A_num_rows, dY, HIP_R_32F);

    hipsparseSpMV_bufferSize(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, vecX, &beta, vecY, HIP_R_32F,
                            HIPSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    hipMalloc(&dBuffer, bufferSize);

    hipsparseSpMV(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, HIP_R_32F,
                                 HIPSPARSE_SPMV_ALG_DEFAULT, dBuffer);


    hipsparseDestroySpMat(matA);
    hipsparseDestroyDnVec(vecX);
    hipsparseDestroyDnVec(vecY);
    hipsparseDestroy(handle);

    hipMemcpy(hY, dY, A_num_rows * sizeof(float),
                           hipMemcpyDeviceToHost);


    int correct = 1;
    for (int i = 0; i < A_num_rows; i++) {
        if (hY[i] != hY_result[i]) { // direct floating point comparison is not
            correct = 0;             // reliable
            break;
        }
    }
    if (correct)
        printf("spmv_csr_example test PASSED\n");
    else
        printf("spmv_csr_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    hipFree(dBuffer);
    hipFree(dA_csrOffsets);
    hipFree(dA_columns);
    hipFree(dA_values);
    hipFree(dX);
    hipFree(dY);
    return EXIT_SUCCESS;
}
