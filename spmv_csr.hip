#include <hip/hip_runtime.h>
#include <cstdio>
#include <hip/hip_runtime_api.h>
#include <hipsparse.h>
#include <vector>
#include <array>
#include <iostream>
#include <hip/hip_complex.h> // Include HIP complex number support

// Function to compare two complex numbers with a given tolerance
bool hipCmplxEquals(hipFloatComplex a, hipFloatComplex b, float tol = 1e-6) {
    return fabs(hipCrealf(a) - hipCrealf(b)) < tol && fabs(hipCimagf(a) - hipCimagf(b)) < tol;
}

int main(void) {
    // Host problem definition for complex numbers
    const int A_num_rows      = 4;
    const int A_num_cols      = 4;
    const int A_nnz           = 9;
    int       hA_csrOffsets[] = { 0, 3, 4, 7, 9 };
    int       hA_columns[]    = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
    hipFloatComplex hA_values[] = { // Complex numbers with 0 imaginary parts
        make_hipFloatComplex(1.0f, 0.0f),
        make_hipFloatComplex(2.0f, 0.0f),
        make_hipFloatComplex(3.0f, 0.0f),
        make_hipFloatComplex(4.0f, 0.0f),
        make_hipFloatComplex(5.0f, 0.0f),
        make_hipFloatComplex(6.0f, 0.0f),
        make_hipFloatComplex(7.0f, 0.0f),
        make_hipFloatComplex(8.0f, 0.0f),
        make_hipFloatComplex(9.0f, 0.0f)
    };
    hipFloatComplex hX[] = {
        make_hipFloatComplex(1.0f, 0.0f),
        make_hipFloatComplex(2.0f, 0.0f),
        make_hipFloatComplex(3.0f, 0.0f),
        make_hipFloatComplex(4.0f, 0.0f)
    };
    hipFloatComplex hY[] = {
        make_hipFloatComplex(0.0f, 0.0f),
        make_hipFloatComplex(0.0f, 0.0f),
        make_hipFloatComplex(0.0f, 0.0f),
        make_hipFloatComplex(0.0f, 0.0f)
    };
    hipFloatComplex alpha = make_hipFloatComplex(1.0f, 0.0f);
    hipFloatComplex beta = make_hipFloatComplex(0.0f, 0.0f);

    // Device memory management for complex numbers
    int            *dA_csrOffsets, *dA_columns;
    hipFloatComplex *dA_values, *dX, *dY;
    hipMalloc((void **)&dA_csrOffsets, (A_num_rows + 1) * sizeof(int));
    hipMalloc((void **)&dA_columns, A_nnz * sizeof(int));
    hipMalloc((void **)&dA_values, A_nnz * sizeof(hipFloatComplex));
    hipMalloc((void **)&dX, A_num_cols * sizeof(hipFloatComplex));
    hipMalloc((void **)&dY, A_num_rows * sizeof(hipFloatComplex));

    // Transfer data from host to device
    hipMemcpy(dA_csrOffsets, hA_csrOffsets, (A_num_rows + 1) * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(dA_values, hA_values, A_nnz * sizeof(hipFloatComplex), hipMemcpyHostToDevice);
    hipMemcpy(dX, hX, A_num_cols * sizeof(hipFloatComplex), hipMemcpyHostToDevice);
    hipMemcpy(dY, hY, A_num_rows * sizeof(hipFloatComplex), hipMemcpyHostToDevice);

    // HIPSPARSE APIs
    hipsparseHandle_t     handle = NULL;
    hipsparseSpMatDescr_t matA;
    hipsparseDnVecDescr_t vecX, vecY;
    void*                 dBuffer    = NULL;
    size_t                bufferSize = 0;

    hipsparseCreate(&handle);
    hipsparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,                        dA_csrOffsets, dA_columns, dA_values,
                        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                        HIPSPARSE_INDEX_BASE_ZERO, HIP_C_32F);

    // Create dense vector X
    hipsparseCreateDnVec(&vecX, A_num_cols, dX, HIP_C_32F);
    // Create dense vector Y
    hipsparseCreateDnVec(&vecY, A_num_rows, dY, HIP_C_32F);

    hipsparseSpMV_bufferSize(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, vecX, &beta, vecY, HIP_C_32F,
                            HIPSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    hipMalloc(&dBuffer, bufferSize);

    hipsparseSpMV(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, HIP_C_32F,
                                 HIPSPARSE_SPMV_ALG_DEFAULT, dBuffer);

    // Wait for the computation to finish
    hipDeviceSynchronize();

    // Copy the result back to the host
    hipMemcpy(hY, dY, A_num_rows * sizeof(hipFloatComplex), hipMemcpyDeviceToHost);

    // Check the result
    hipFloatComplex hY_result[] = { // Update with your expected complex result
        make_hipFloatComplex(19.0f, 0.0f),
        make_hipFloatComplex(8.0f, 0.0f),
        make_hipFloatComplex(51.0f, 0.0f),
        make_hipFloatComplex(52.0f, 0.0f)
    };
    int correct = 1;
    for (int i = 0; i < A_num_rows; i++) {
        if (!hipCmplxEquals(hY[i], hY_result[i])) { // Use complex number comparison
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("spmv_csr_example test PASSED\n");
    else
        printf("spmv_csr_example test FAILED: wrong result\n");

    // Clean up
    hipsparseDestroySpMat(matA);
    hipsparseDestroyDnVec(vecX);
    hipsparseDestroyDnVec(vecY);
    hipsparseDestroy(handle);

    hipFree(dBuffer);
    hipFree(dA_csrOffsets);
    hipFree(dA_columns);
    hipFree(dA_values);
    hipFree(dX);
    hipFree(dY);

    return EXIT_SUCCESS;
}
