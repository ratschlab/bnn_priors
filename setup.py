from setuptools import setup

setup(
    name="bnn_priors",
    version="0.1.0",
    description="investigations on BNN priors",
    long_description="",
    author="Vincent Fortuin, Adrià Garriga-Alonso",
    author_email="adria.garriga@gmail.com",
    url="https://github.com/ratschlab/projects2020_BNN-priors/",
    license="MIT license",
    packages=["bnn_priors"],
    install_requires=[
        "torch>=1.5.0<1.6",
        "tensorboardX>=2.0<2.1",
        "sacred>=0.8<0.9",
        "gpytorch>=1.0<1.2",
    ],
    test_suite="testing",
)
