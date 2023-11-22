"""Setup file"""
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "pylord.cythonized",            # Change this to your module name
        sources=["pylord/cythonized.pyx"],
        include_dirs=[np.get_include()],
        language="c"
    )
]


setup(
    name='pylord',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=cythonize(extensions)
)

