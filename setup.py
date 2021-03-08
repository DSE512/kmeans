import numpy as np
from Cython.Build import cythonize
from setuptools import find_packages, setup, Extension


with open('README.md') as readme_file:
    readme = readme_file.read()


setup_requirements = ['pytest-runner', ]
test_requirements = ['pytest>=3', ]
requirements = ['cython']


extensions = [
    Extension(
        name='cluster._cluster_means',
        sources=['cluster/_cluster_means.pyx']
    )
]


setup(
    author="Todd Young",
    author_email='youngmt1@ornl.gov',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="K-means",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='kmeans',
    name='cluster',
    packages=find_packages(include=['cluster', 'cluster.*']),
    include_dirs = [np.get_include()],
    ext_modules=cythonize(extensions),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/DSE512/kmeans',
    version='0.1.0',
    zip_safe=False,
)
