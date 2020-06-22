from setuptools import setup
from setuptools import find_packages

long_description = '''
Libra is a machine learning API that makes building and deploying models as simple as a one-line function call. 
'''

setup(name='libra',
      version='0.0.0',
      description='Deep Learning in fluent one-liners',
      long_description=long_description,
      author='Palash Shah',
      author_email='ps9cmk@virginia.edu',
      url='https://github.com/Palashio/libra',
      license='MIT',
      install_requires=['numpy>=1.9.1',
                        'scipy>=0.14',
                        'pyyaml',
                        'h5py'],
      extras_require={
          'visualize': ['pydot>=1.2.4'],
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist',
                    'flaky',
                    'pytest-cov',
                    'pandas',
                    'requests',
                    'markdown'],
      },
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())