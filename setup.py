"""
Setup script for Modern Diffuser package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="m_diffuser",
    version="0.1.0",
    author="Your Name",
    description="Modern implementation of Diffuser for trajectory planning with Gymnasium",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/m_diffuser",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "gymnasium>=1.0.0",
        "gymnasium-robotics>=1.3.0",
        "minari>=0.4.0",
        "mujoco>=3.1.0,<3.2.0",
        "einops>=0.6.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "ipython>=8.0.0",
            "jupyter>=1.0.0",
        ],
        "logging": [
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "m-diffuser-train=scripts.train:main",
            "m-diffuser-eval=scripts.evaluate:main",
            "m-diffuser-download=scripts.download_data:main",
        ],
    },
)