"""Build the curriculum_tokenizer C++ extension."""

import os
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

if os.name == "nt":
    compile_args = ["/O2", "/DNDEBUG"]
else:
    compile_args = ["-O3", "-DNDEBUG", "-pthread"]

ext_modules = [
    Pybind11Extension(
        "curriculum_tokenizer._cpp_tokenizer",
        ["src/bindings.cpp", "src/tokenizer.cpp"],
        include_dirs=["include"],
        cxx_std=17,
        extra_compile_args=compile_args,
        extra_link_args=[] if os.name == "nt" else ["-pthread"],
    ),
]

setup(
    name="bloat-tokenizer",
    version="0.3.0",
    description="Lossless C++ BPE tokenizer and dataset token-counting CLI",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    entry_points={"console_scripts": ["bloat-tokenizer=curriculum_tokenizer.cli:main"]},
    zip_safe=False,
    python_requires=">=3.8",
)
