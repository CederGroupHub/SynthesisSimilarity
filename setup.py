from setuptools import setup, find_packages


__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He'
__email__ = 'tanjin_he@berkeley.edu'


if __name__ == "__main__":
    setup(name='SynthesisSimilarity',
          version='1.0.0',
          author="Tanjin He",
          author_email="tanjin_he@berkeley.edu",
          license="MIT License",
          packages=find_packages(),
          include_package_data=True,
          install_requires=[
              'pymatgen',
              'adjustText',
              'colorcet',
              'psutil',
              'seaborn',
              'jsonlines',
              'matplotlib',
              'tensorflow==2.7.0',
              'protobuf==3.19.6',
              'regex',
              'timebudget',
              'scikit-learn',
              'gdown',
          ],
          zip_safe=False)

