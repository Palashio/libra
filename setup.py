import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="verve", # Replace with your own username
    version="0.0.1",
    author="Palash Shah",
    author_email="ps9cmk@virginia.edu",
    description="Fully automated machine learning in one-liners.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Palashio/verve",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)