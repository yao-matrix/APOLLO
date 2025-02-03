from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="apollo-torch",
    version="1.0.2",
    description="APOLLO: SGD-like Memory, AdamW-level Performance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhuhanqing/APOLLO",
    author="Hanqing Zhu",
    author_email="hqzhu@utexas.edu",
    license="CC-BY-NC",  # Specify the license
    packages=["apollo_torch"],
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
