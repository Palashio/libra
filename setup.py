import setuptools
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="libra",  # Replace with your own username
    version="0.1.0",
    author="Palash Shah",
    author_email="ps9cmk@virginia.edu",
    description="Fully automated machine learning in one-liners.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Palashio/libra",
    install_requires=['colorama',
                      'transformers==2.11.0',
                      'tensorflow',
                      'keras',
                      'numpy',
                      'pandas',
                      'sklearn',
                      'pprint',
                      'matplotlib',
                      'tabulate',
                      'textblob',
                      'python-Levenshtein',
                      'seaborn',
                      'keras-tuner',
                      'spacy',
                      'torch',
                      'autocorrect',
                      'pillow',
                      'prince',
                      'opencv-python',
                      'nltk'],
    packages=find_packages(exclude=('tests',)),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
