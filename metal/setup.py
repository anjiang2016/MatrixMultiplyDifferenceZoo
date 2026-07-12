from setuptools import setup, Extension
import numpy as np

ext = Extension(
    'convmodule',
    sources=[
        'convmodule.c',
        'metal_runtime.m'
    ],
    include_dirs=[np.get_include()],
    extra_compile_args=[
        '-O0',
        '-g',
        '-Xclang', '-fopenmp'
    ],
    extra_link_args=[
        '-framework', 'Metal',
        '-framework', 'Foundation',
        '-framework', 'MetalPerformanceShaders',  # 添加 MPS 框架
        '-lomp'
    ]
)

setup(name='convmodule', ext_modules=[ext])
