from setuptools import setup, Extension
import numpy as np
import os

os.environ['CC'] = 'clang'
os.environ['CXX'] = 'clang++'

ext = Extension(
    'convmodule',
    sources=['convmodule.c'],
    include_dirs=[np.get_include()],
    extra_compile_args=['-O3', '-fopenmp', '-march=native', '-ffast-math'],
    extra_link_args=['-fopenmp'],
)

setup(
    name='convmodule',
    ext_modules=[ext],
    #zip_safe=False,
)
