import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hierarch",
    version="1.1.1",
    author="Rishi Kulkarni",
    author_email="rkulk@stanford.edu",
    description="Hierarchical hypothesis testing library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license_files=("LICENSE.txt",),
    url="https://github.com/rishi-kulkarni/hierarch",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy>=1.20.2",
        "scipy>=1.6.2",
        "numba>=0.53.1",
        "pandas>=1.2.4",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)

