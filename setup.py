import setuptools
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="libra",  # Replace with your own username
    version="1.0.3",
    author="Palash Shah",
    author_email="ps9cmk@virginia.edu",
    description="Fully automated machine learning in one-liners.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Palashio/libra",
    install_requires=['colorama',
                      'transformers==2.11.0',
                      'tensorflow==2.2.0',
                      'keras==2.4.3',
                      'numpy',
                      'pandas',
                      'sklearn',
                      'pprint',
                      'matplotlib',
                      'tabulate',
                      'textblob',
                      'seaborn',
                      'keras-tuner',
                      'spacy',
                      'jellyfish',
                      'autocorrect',
                      'pillow',
                      'prince',
                      'opencv-python',
                      'nltk',
                      'xgboost'],
    packages=find_packages(exclude=('tests',)),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
