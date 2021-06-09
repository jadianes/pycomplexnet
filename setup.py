from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='pyComplexNet',
    version='0.1',
    description='Complex Networks in Python',
    long_description=readme(),
    keywords='complex networks graphs mathematics',
    url='http://github.com/jadianes/pycomplexnet',
    author='Jose A. Dianes',
    author_email='jadianes@gmail.com',
    license='MIT',
    packages=['pycomplexnet'],
    install_requires=[
        'numpy',
        'scipy'
    ],
    include_package_data=True,
    zip_safe=False,
    test_suite='nose.collector',
    tests_require=['nose'])