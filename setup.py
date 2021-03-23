from os import path

import setuptools
from setuptools import setup

extras = {
    'test': ['pytest', 'pytest_cases'],
}
# Meta dependency groups.
extras['all'] = [item for group in extras.values() for item in group]

setup(name='policybazaar',
      version='0.0.1-alpha1',
      description='A collection of different quality policies for reinforcement learning .',
      long_description_content_type='text/markdown',
      long_description=open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf-8').read(),
      url='https://github.com/koulanurag/policybazaar',
      author='Anurag Koul',
      author_email='koulanurag@gmail.com',
      license=open(path.join(path.abspath(path.dirname(__file__)), 'LICENSE'), encoding='utf-8').read(),
      packages=setuptools.find_packages(),
      install_requires=['wandb>=0.10',
                        'gym>=0.17.0',
                        'torch>=1.8.0',
                        "d4rl @ git+https://git@github.com/rail-berkeley/d4rl@master#egg=d4rl"],
      extras_require=extras,
      tests_require=extras['test'],
      python_requires='>=3.5',
      classifiers=[
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
      ],
      )
