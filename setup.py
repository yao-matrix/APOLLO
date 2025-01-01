from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="apollo-torch",
    version="1.0",
    description="APOLLO: SGD-like Memory, AdamW-level Performance",
    url="https://github.com/zhuhanqing/APOLLO",
    author="Hanqing Zhu",
    author_email="hqzhu@utexas.edu",
    license="Apache 2.0",
    packages=["apollo_torch"],
    install_requires=required,
)