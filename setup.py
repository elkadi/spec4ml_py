from setuptools import setup, find_packages

setup(
    name="spec4ml_py",
    version="0.1.0",
    packages=find_packages(where="spec4ml_py"),
    package_dir={"": "spec4ml_py"},
    author="Omar Anwar Elkadi",
    description="Python package for handling and analyzing spectral data",
    url="https://github.com/elkadi/spec4ml_py",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", 
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
