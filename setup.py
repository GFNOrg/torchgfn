from setuptools import find_packages, setup

setup(
    name="gfn",
    version="0.2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["numpy", "torch", "torchtyping", "einops", "gymnasium"],
)
