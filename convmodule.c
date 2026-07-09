// convmodule.c - 纯 C 实现的卷积（带 OpenMP 并行）
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <omp.h>
#include <string.h>

// -------------------- 卷积前向（C + OpenMP）--------------------
static PyObject* conv_c(PyObject* self, PyObject* args) {
    PyArrayObject *x, *w;
    int stride, padding;

    if (!PyArg_ParseTuple(args, "O!O!ii", &PyArray_Type, &x,
                          &PyArray_Type, &w, &stride, &padding)) {
        return NULL;
    }

    // 确保输入是 C 连续（行优先）
    if (!PyArray_IS_C_CONTIGUOUS(x)) {
        PyArrayObject *x_contig = (PyArrayObject*)PyArray_FromAny(
            (PyObject*)x, PyArray_DescrFromType(NPY_FLOAT32), 0, 0,
            NPY_ARRAY_C_CONTIGUOUS, NULL);
        if (!x_contig) return NULL;
        x = x_contig;
    }
    if (!PyArray_IS_C_CONTIGUOUS(w)) {
        PyArrayObject *w_contig = (PyArrayObject*)PyArray_FromAny(
            (PyObject*)w, PyArray_DescrFromType(NPY_FLOAT32), 0, 0,
            NPY_ARRAY_C_CONTIGUOUS, NULL);
        if (!w_contig) return NULL;
        w = w_contig;
    }

    // 获取维度
    npy_intp N = PyArray_DIM(x, 0);
    npy_intp C = PyArray_DIM(x, 1);
    npy_intp H = PyArray_DIM(x, 2);
    npy_intp W = PyArray_DIM(x, 3);
    npy_intp C_out = PyArray_DIM(w, 0);
    npy_intp Hk = PyArray_DIM(w, 2);
    npy_intp Wk = PyArray_DIM(w, 3);

    // 检查卷积核是否能放入输入
    if (Hk > H + 2*padding || Wk > W + 2*padding) {
        PyErr_SetString(PyExc_ValueError, "Kernel larger than padded input");
        return NULL;
    }

    npy_intp H_out = (H + 2*padding - Hk) / stride + 1;
    npy_intp W_out = (W + 2*padding - Wk) / stride + 1;

    npy_intp dims[] = {N, C_out, H_out, W_out};
    PyArrayObject* out = (PyArrayObject*)PyArray_SimpleNew(4, dims, NPY_FLOAT32);
    if (!out) return NULL;

    float* x_ptr = (float*)PyArray_DATA(x);
    float* w_ptr = (float*)PyArray_DATA(w);
    float* out_ptr = (float*)PyArray_DATA(out);

    // 使用 OpenMP 并行（加保护）
    #pragma omp parallel for collapse(4) schedule(static)
    //#pragma omp parallel for collapse(4) schedule(dynamic,1)
    for (npy_intp n = 0; n < N; n++) {
        for (npy_intp co = 0; co < C_out; co++) {
            for (npy_intp ho = 0; ho < H_out; ho++) {
                for (npy_intp wo = 0; wo < W_out; wo++) {
                    float sum = 0.0f;
                    for (npy_intp ci = 0; ci < C; ci++) {
                        for (npy_intp kh = 0; kh < Hk; kh++) {
                            for (npy_intp kw = 0; kw < Wk; kw++) {
                                npy_intp h_idx = ho * stride + kh - padding;
                                npy_intp w_idx = wo * stride + kw - padding;
                                // 边界检查
                                if (h_idx >= 0 && h_idx < H && w_idx >= 0 && w_idx < W) {
                                    npy_intp x_idx = n*C*H*W + ci*H*W + h_idx*W + w_idx;
                                    npy_intp w_idx_c = co*C*Hk*Wk + ci*Hk*Wk + kh*Wk + kw;
                                    sum += x_ptr[x_idx] * w_ptr[w_idx_c];
                                }
                            }
                        }
                    }
                    npy_intp out_idx = n*C_out*H_out*W_out + co*H_out*W_out + ho*W_out + wo;
                    out_ptr[out_idx] = sum;
                }
            }
        }
    }

    return (PyObject*)out;
}

// ============================================================
// 卷积反向传播 (C + OpenMP)
// ============================================================
static PyObject* d_conv_c(PyObject* self, PyObject* args) {
    PyArrayObject *dout, *x, *w;
    int stride = 1, padding = 0;
    PyArrayObject *x_pad = NULL;

    // 只解析 6 个参数
    if (!PyArg_ParseTuple(args, "O!O!O!iiO!",
                          &PyArray_Type, &dout,
                          &PyArray_Type, &x,
                          &PyArray_Type, &w,
                          &stride, &padding,
                          &PyArray_Type, &x_pad)) {
        return NULL;
    }

    npy_intp N = PyArray_DIM(x, 0);
    npy_intp C = PyArray_DIM(x, 1);
    npy_intp H = PyArray_DIM(x, 2);
    npy_intp W = PyArray_DIM(x, 3);
    npy_intp C_out = PyArray_DIM(w, 0);
    npy_intp Hk = PyArray_DIM(w, 2);
    npy_intp Wk = PyArray_DIM(w, 3);

    // 内部计算 H_out 和 W_out
    npy_intp H_out = (H + 2*padding - Hk) / stride + 1;
    npy_intp W_out = (W + 2*padding - Wk) / stride + 1;

    if (H_out == -1) {
        H_out = (H + 2*padding - Hk) / stride + 1;
        W_out = (W + 2*padding - Wk) / stride + 1;
    }

    npy_intp dx_dims[] = {N, C, H, W};
    npy_intp dw_dims[] = {C_out, C, Hk, Wk};
    npy_intp db_dims[] = {C_out};

    PyArrayObject* dx = (PyArrayObject*)PyArray_SimpleNew(4, dx_dims, NPY_FLOAT32);
    PyArrayObject* dw = (PyArrayObject*)PyArray_SimpleNew(4, dw_dims, NPY_FLOAT32);
    PyArrayObject* db = (PyArrayObject*)PyArray_SimpleNew(1, db_dims, NPY_FLOAT32);

    if (!dx || !dw || !db) {
        Py_XDECREF(dx); Py_XDECREF(dw); Py_XDECREF(db);
        return NULL;
    }

    float* dout_ptr = (float*)PyArray_DATA(dout);
    float* x_ptr = (float*)PyArray_DATA(x);
    float* w_ptr = (float*)PyArray_DATA(w);
    float* dx_ptr = (float*)PyArray_DATA(dx);
    float* dw_ptr = (float*)PyArray_DATA(dw);
    float* db_ptr = (float*)PyArray_DATA(db);

    memset(dx_ptr, 0, N*C*H*W * sizeof(float));
    memset(dw_ptr, 0, C_out*C*Hk*Wk * sizeof(float));
    memset(db_ptr, 0, C_out * sizeof(float));

    // ---------- 计算 db (OpenMP 并行，原子累加) ----------
    #pragma omp parallel for collapse(2) schedule(static)
    //#pragma omp parallel for collapse(2) schedule(dynamic,1)
    for (npy_intp n = 0; n < N; n++) {
        for (npy_intp co = 0; co < C_out; co++) {
            float sum = 0.0f;
            for (npy_intp ho = 0; ho < H_out; ho++) {
                for (npy_intp wo = 0; wo < W_out; wo++) {
                    sum += dout_ptr[n*C_out*H_out*W_out + co*H_out*W_out + ho*W_out + wo];
                }
            }
            #pragma omp atomic
            db_ptr[co] += sum;
        }
    }

    // ---------- 计算 dw 和 dx (OpenMP 并行，原子累加) ----------
    #pragma omp parallel for collapse(4) schedule(static)
    //#pragma omp parallel for collapse(4) schedule(dynamic,1)
    for (npy_intp n = 0; n < N; n++) {
        for (npy_intp co = 0; co < C_out; co++) {
            for (npy_intp ho = 0; ho < H_out; ho++) {
                for (npy_intp wo = 0; wo < W_out; wo++) {
                    float d = dout_ptr[n*C_out*H_out*W_out + co*H_out*W_out + ho*W_out + wo];
                    if (d == 0.0f) continue;

                    npy_intp h_start = ho * stride - padding;
                    npy_intp w_start = wo * stride - padding;

                    for (npy_intp ci = 0; ci < C; ci++) {
                        for (npy_intp kh = 0; kh < Hk; kh++) {
                            for (npy_intp kw = 0; kw < Wk; kw++) {
                                npy_intp h_idx = h_start + kh;
                                npy_intp w_idx = w_start + kw;
                                if (h_idx >= 0 && h_idx < H && w_idx >= 0 && w_idx < W) {
                                    npy_intp x_idx = n*C*H*W + ci*H*W + h_idx*W + w_idx;
                                    float x_val = x_ptr[x_idx];
                                    float w_val = w_ptr[co*C*Hk*Wk + ci*Hk*Wk + kh*Wk + kw];

                                    npy_intp dw_idx = co*C*Hk*Wk + ci*Hk*Wk + kh*Wk + kw;
                                    npy_intp dx_idx = n*C*H*W + ci*H*W + h_idx*W + w_idx;

                                    #pragma omp atomic
                                    dw_ptr[dw_idx] += x_val * d;
                                    #pragma omp atomic
                                    dx_ptr[dx_idx] += w_val * d;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    PyObject* result = PyTuple_New(3);
    PyTuple_SetItem(result, 0, (PyObject*)dx);
    PyTuple_SetItem(result, 1, (PyObject*)dw);
    PyTuple_SetItem(result, 2, (PyObject*)db);
    return result;
}

// -------------------- 模块初始化 --------------------
static PyMethodDef ConvMethods[] = {
    {"conv_c", conv_c, METH_VARARGS, "Fast C implementation of convolution."},
    {"d_conv_c", d_conv_c, METH_VARARGS, "Fast C implementation of convolution backward.2259"},
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
    import_array();
    return PyModule_Create(&convmodule);
}
