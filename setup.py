from setuptools import setup, find_packages

setup(
    name="riemannML",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "torch", "torchdiffeq", "numpy", "matplotlib"
    ]
)
