#!/usr/bin/env python
# The code framework of WCCNet is extended from YOLOX (https://github.com/Megvii-BaseDetection/YOLOX)

import re
import setuptools
import glob
from os import path
import torch
from torch.utils.cpp_extension import CppExtension


def get_extensions():
    # get abspath of the root dir for wccnet repos
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "wccnet", "layers", "csrc")

    main_source = path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))

    sources = [main_source] + sources
    extension = CppExtension

    extra_compile_args = {"cxx": ["-O3"]}
    define_macros = []

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "wccnet._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


def get_package_dir():
    pkg_dir = {
        "wccnet.tools": "wccnet/tools",
        "wccnet.exp.default": "wccnet/exps/default",
    }
    return pkg_dir


def get_install_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        reqs = [x.strip() for x in f.read().splitlines()]
    reqs = [x for x in reqs if not x.startswith("#")]
    return reqs


def get_wccnet_version():
    with open("wccnet/__init__.py", "r") as f:
        version = re.search(
            r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
            f.read(), re.MULTILINE
        ).group(1)
    return version


def get_long_description():
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
    return long_description

setuptools.setup(
    name="wccnet",
    version=get_wccnet_version(),
    author="Cyber Pegasus",
    url="https://github.com/CyberPegasus/WCCNet",
    package_dir=get_package_dir(),
    python_requires=">=3.7",
    install_requires=get_install_requirements(),
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    ext_modules=get_extensions(),
    classifiers=[
        "Programming Language :: Python :: 3", "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
    ],
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    packages=setuptools.find_packages(),
    project_urls={
        "Source": "https://github.com/CyberPegasus/WCCNet",
        "Tracker": "https://github.com/CyberPegasus/WCCNet/issues",
    },
)
