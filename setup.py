import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name="hierarch",
                 version="0.2.1",

                 author="Rishi Kulkarni",
                 author_email="rkulk@stanford.edu",
                 description="Hierarchical hypothesis testing library",
                 long_description=long_description,
                 long_description_content_type="text/markdown",

                 license_files=('LICENSE.txt',),

                 url="https://github.com/rishi-kulkarni/hierarch",


                 packages=setuptools.find_packages(),
                 install_requires=["numpy", "sympy", "scipy",
                                   "numba", "pandas"],
                 classifiers=[
                    "Programming Language :: Python :: 3",
                    "License :: OSI Approved :: MIT License",
                    "Operating System :: OS Independent"
                 ],
                 python_requires='>=3.7'
                 )