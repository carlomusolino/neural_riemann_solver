from setuptools import setup, find_packages

setup(
    name="riemannML",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "torch", "torchdiffeq", "numpy", "matplotlib"
    ],
    entry_points={
        'console_scripts': [
            'train-models=riemannML.cli.train_models_script:main',
            'run-sims=riemannML.cli.run_sims_script:main',
            'run-sims-ensemble=riemannML.cli.run_sims_ensemble_script:main',
        ],
    }
)

