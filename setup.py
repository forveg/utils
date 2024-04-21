from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="forveg utils",
    version="0.0.1",
    author="forveg",
    description="A bunch of utils I typically use",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.2.2",
        "numpy>=1.24.4",
        "matplotlib>=3.8.4",
        "scikit-learn>=1.4.2",
        "lightgbm>=4.3.0"
    ],
    extras_require={
        "preproc": ["tsfresh>=0.20.1"],
    },
    include_package_data=False,
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
)