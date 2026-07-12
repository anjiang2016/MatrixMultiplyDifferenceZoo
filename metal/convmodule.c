#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>

// 声明 Metal 函数
void metal_init(void);
void metal_matmul(const float* A, const float* B, float* C,
                  unsigned int M, unsigned int N, unsigned int K,
                  unsigned int* elapsed_ms);
int metal_is_available(void);

// ============ 测试函数 ============
static PyObject* test_matmul(PyObject* self, PyObject* args) {
    printf("[LOG] test_matmul: entered\n");
    fflush(stdout);

    PyArrayObject *A, *B;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &A, &PyArray_Type, &B)) {
        printf("[LOG] test_matmul: argument parsing failed\n");
        fflush(stdout);
        return NULL;
    }
    printf("[LOG] test_matmul: arguments parsed\n");
    fflush(stdout);

    npy_intp* dimsA = PyArray_DIMS(A);
    npy_intp* dimsB = PyArray_DIMS(B);
    printf("[LOG] test_matmul: A shape (%ld, %ld), B shape (%ld, %ld)\n",
           dimsA[0], dimsA[1], dimsB[0], dimsB[1]);
    fflush(stdout);

    if (PyArray_NDIM(A) != 2 || PyArray_NDIM(B) != 2) {
        PyErr_SetString(PyExc_ValueError, "输入必须是二维矩阵");
        printf("[LOG] test_matmul: dimension error\n");
        fflush(stdout);
        return NULL;
    }
    unsigned int M = dimsA[0];
    unsigned int K = dimsA[1];
    unsigned int N = dimsB[1];
    if (dimsB[0] != K) {
        PyErr_SetString(PyExc_ValueError, "矩阵维度不匹配");
        printf("[LOG] test_matmul: dimension mismatch\n");
        fflush(stdout);
        return NULL;
    }

    float* A_ptr = (float*)PyArray_DATA(A);
    float* B_ptr = (float*)PyArray_DATA(B);
    printf("[LOG] test_matmul: A_ptr=%p, B_ptr=%p\n", A_ptr, B_ptr);
    fflush(stdout);

    npy_intp dimsC[] = {M, N};
    PyArrayObject* C = (PyArrayObject*)PyArray_SimpleNew(2, dimsC, NPY_FLOAT32);
    if (!C) {
        printf("[LOG] test_matmul: failed to create output array\n");
        fflush(stdout);
        return NULL;
    }
    float* C_ptr = (float*)PyArray_DATA(C);
    printf("[LOG] test_matmul: output array created, C_ptr=%p\n", C_ptr);
    fflush(stdout);

    unsigned int elapsed_ms = 0;
    int available = metal_is_available();
    printf("[LOG] test_matmul: metal_is_available() = %d\n", available);
    fflush(stdout);

    if (available) {
        printf("[LOG] test_matmul: calling metal_matmul...\n");
        fflush(stdout);
        metal_matmul(A_ptr, B_ptr, C_ptr, M, N, K, &elapsed_ms);
        printf("[LOG] test_matmul: metal_matmul returned, elapsed_ms=%u\n", elapsed_ms);
        fflush(stdout);
    } else {
        printf("[LOG] test_matmul: falling back to CPU\n");
        fflush(stdout);
        // CPU 回退（简单三重循环）
        for (unsigned int i = 0; i < M; i++) {
            for (unsigned int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (unsigned int k = 0; k < K; k++) {
                    sum += A_ptr[i * K + k] * B_ptr[k * N + j];
                }
                C_ptr[i * N + j] = sum;
            }
        }
        elapsed_ms = 0;
    }

    PyObject* result = PyTuple_New(2);
    PyTuple_SetItem(result, 0, (PyObject*)C);
    PyTuple_SetItem(result, 1, PyLong_FromUnsignedLong(elapsed_ms));
    printf("[LOG] test_matmul: returning result\n");
    fflush(stdout);
    return result;
}

// ============ 模块方法定义 ============
static PyMethodDef ConvMethods[] = {
    {"test_matmul", test_matmul, METH_VARARGS,
     "Test matrix multiplication with Metal."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef convmodule = {
    PyModuleDef_HEAD_INIT,
    "convmodule",
    NULL,
    -1,
    ConvMethods
};

PyMODINIT_FUNC PyInit_convmodule(void) {
    printf("[LOG] PyInit_convmodule: entering\n");
    fflush(stdout);
    import_array();
    printf("[LOG] PyInit_convmodule: import_array done\n");
    fflush(stdout);

    // 这里先不调用 metal_init，避免模块加载时崩溃
    // 改为在 test_matmul 中按需初始化
    printf("[LOG] PyInit_convmodule: returning module\n");
    fflush(stdout);
    return PyModule_Create(&convmodule);
}
