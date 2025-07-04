#!/usr/bin/env python3
"""
Central Bank Speech Analysis Platform - Package Setup

Production-ready package configuration for the Central Bank Speech Analysis Platform.
Follows best practices for Python packaging and distribution.

Author: Central Bank Speech Analysis Platform
Date: 2025
Version: 2.0.0
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    requirements_path = this_directory / filename
    if requirements_path.exists():
        return requirements_path.read_text().strip().split('\n')
    return []

install_requires = read_requirements("requirements.txt")
dev_requires = read_requirements("requirements-dev.txt")

setup(
   name="central-bank-speech-platform",
   version="2.0.0",
   description="Production-grade platform for collecting and analyzing central bank speeches",
   long_description=long_description,
   long_description_content_type="text/markdown",
   author="Central Bank Speech Analysis Platform Team",
   author_email="admin@centralbankspeech.com",
   url="https://github.com/centralbankspeech/platform",
   
   # Package configuration
   packages=find_packages(exclude=["tests*", "testing*"]),
   python_requires=">=3.11",
   install_requires=install_requires,
   extras_require={
       "dev": dev_requires,
       "selenium": ["selenium>=4.15.0"],
       "redis": ["redis>=4.5.0"],
       "elasticsearch": ["elasticsearch>=8.0.0"],
       "all": dev_requires + ["selenium>=4.15.0", "redis>=4.5.0", "elasticsearch>=8.0.0"],
   },
   
   # Entry points
   entry_points={
       "console_scripts": [
           "speechctl=tools.cli:main",
           "cb-speech=tools.cli:main",
       ],
       "central_bank_plugins": [
           "fed=plugins.federal_reserve.plugin:FederalReservePlugin",
           "boe=plugins.bank_of_england.plugin:BankOfEnglandPlugin",
           "bis=plugins.bis.plugin:BISPlugin",
       ],
   },
   
   # Package metadata
   classifiers=[
       "Development Status :: 4 - Beta",
       "Intended Audience :: Developers",
       "Intended Audience :: Financial and Insurance Industry",
       "Intended Audience :: Science/Research",
       "License :: OSI Approved :: MIT License",
       "Operating System :: OS Independent",
       "Programming Language :: Python :: 3",
       "Programming Language :: Python :: 3.11",
       "Programming Language :: Python :: 3.12",
       "Topic :: Office/Business :: Financial",
       "Topic :: Scientific/Engineering :: Information Analysis",
       "Topic :: Text Processing :: Linguistic",
       "Typing :: Typed",
   ],
   
   # Keywords for PyPI
   keywords=[
       "central-bank", "monetary-policy", "speech-analysis", "nlp", 
       "financial-data", "federal-reserve", "scraping", "economics",
       "sentiment-analysis", "policy-analysis", "ddd", "async"
   ],
   
   # Package data
   include_package_data=True,
   package_data={
       "config": ["*.yaml", "*.yml"],
       "nlp": ["models/*", "lexicons/*"],
       "plugins": ["*/data/*"],
   },
   
   # Project URLs
   project_urls={
       "Bug Reports": "https://github.com/centralbankspeech/platform/issues",
       "Source": "https://github.com/centralbankspeech/platform",
       "Documentation": "https://centralbankspeech.readthedocs.io/",
       "Changelog": "https://github.com/centralbankspeech/platform/blob/main/CHANGELOG.md",
   },
   
   # Zip safety
   zip_safe=False,
   
   # Test configuration
   test_suite="testing",
   tests_require=[
       "pytest>=7.4.0",
       "pytest-asyncio>=0.21.0",
       "pytest-cov>=4.1.0",
       "pytest-mock>=3.12.0",
   ],
)