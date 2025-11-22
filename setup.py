import setuptools



setuptools.setup(
    name='pyMatan',
    version='0.1.0',
    author="fast make a rgr",
    description='A Python package for numerical and symbolic mathematical analysis.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/pyMatan/pyMatan',


    packages=setuptools.find_packages(exclude=['*tests*']),

    install_requires=[
        'numpy>=1.20',
        'sympy>=1.8',
        'scipy>=1.7',
        'matplotlib>=3.4',
    ],

    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Education',
    ],
    python_requires='>=3.8',
)