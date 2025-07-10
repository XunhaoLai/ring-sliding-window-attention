#!/usr/bin/env python3

from setuptools import setup, find_packages

# Read the contents of README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ring-sliding-window-attention",
    version="0.1.0",
    author="Xunhao Lai",
    author_email="laixunhao@pku.edu.cn",
    description="Ring attention implementation for sliding window attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/XunhaoLai/ring-sliding-window-attention",
    packages=find_packages(include=["ring_swa", "ring_swa.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Only lightweight essential dependencies
        "packaging>=21.0",
    ],
    extras_require={
        "torch": [
            "torch>=2.7.0",
            "einops>=0.6.0",
        ],
        "flash": [
            "flash-attn>=2.5.8",
        ],
        "full": [
            "torch>=2.7.0",
            "flash-attn>=2.5.8",
            "einops>=0.6.0",
            "packaging>=21.0",
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "pre-commit>=2.0",
            "torch>=2.7.0",
            "flash-attn>=2.5.8",
            "einops>=0.6.0",
        ],
        "test": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "torch>=2.7.0",
            "flash-attn>=2.5.8",
            "einops>=0.6.0",
        ],
        "triton": [
            "triton>=3.0.0",
        ],
    },
    keywords=[
        "attention",
        "sliding window",
        "ring attention",
        "flash attention",
        "transformer",
        "pytorch",
        "triton",
    ],
    project_urls={
        "Homepage": "https://github.com/XunhaoLai/ring-sliding-window-attention",
        "Repository": "https://github.com/XunhaoLai/ring-sliding-window-attention",
        "Documentation": "https://github.com/XunhaoLai/ring-sliding-window-attention",
        "Bug Tracker": "https://github.com/XunhaoLai/ring-sliding-window-attention/issues",
    },
)
