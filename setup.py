"""Setup script for Smart Cleaner."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="smart-cleaner",
    version="0.5.0",
    author="Smart Cleaner Team",
    author_email="contact@example.com",
    description="AI-Powered Data Cleaning Tool using Local LLMs (Ollama)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abraham13202/smart-cleaner",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "ollama>=0.3.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "web": [
            "streamlit>=1.29.0",
            "plotly>=5.18.0",
        ],
        "export": [
            "openpyxl>=3.1.0",
            "pyarrow>=14.0.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
        "all": [
            "streamlit>=1.29.0",
            "plotly>=5.18.0",
            "openpyxl>=3.1.0",
            "pyarrow>=14.0.0",
            "python-docx>=0.8.11",
        ],
    },
    entry_points={
        "console_scripts": [
            "smart-cleaner=universal_data_pipeline:main",
        ],
    },
)
