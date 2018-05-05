from setuptools import setup, find_packages

setup(
    name='EEG-Persistent-Homology',
    version='0.0',
    packages=['Holes', 'Holes.classes', 'Holes.drawing', 'Holes.measures', 'Holes.operations', 'Holes.perseus_utils',
              'cython_helpers'],
    url='',
    license='',
    author='Jonathan',
    author_email='lajd@mcmaster.ca',
    description='- Persistent homology on EEG data',
    setup_requires=['numpy==1.13.3', 'cython==0.27.3'],
    install_requires = ['networkx==1.11',
                        'scipy==0.16.1',
                        'matplotlib==2.0.0'],
)



