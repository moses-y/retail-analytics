"""
Setup script for retail analytics package
"""
from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read long description
with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name="retail_analytics",
    version="0.1.0",
    author="Moses Yebei",
    author_email="mosesyebei@gmail.com",
    description="Retail Analytics Platform with Sales Forecasting, Customer Segmentation, and Product Review Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/moses-y/retail-analytics",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "retail-analytics=src.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "retail_analytics": ["config/*.yml"],
    },
)