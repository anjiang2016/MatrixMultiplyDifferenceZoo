

# 获取 Python 的 include 路径（适用于 conda）
PY_INCLUDE=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")
NUMPY_INCLUDE=$(python -c "import numpy; print(numpy.get_include())")
PY_INCLUDE=$(python-config --includes | sed 's/-I//g' | awk '{print $1}')
echo "Using Python include: $PY_INCLUDE"
echo "Using NumPy include: $NUMPY_INCLUDE"


rm -rf __pycache__/ && rm -f convmodule.so &&rm -rf build/
clang -O3 -std=c11 -march=native  -I$NUMPY_INCLUDE   -I$PY_PREFIX/include/python3.9  -shared -o convmodule.so convmodule.c  -undefined dynamic_lookup
python -c "import convmodule; print(dir(convmodule))"
python -c "import convmodule, inspect; print(inspect.signature(convmodule.d_conv_c))"
python test_dconv.py
