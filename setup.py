"""Uses setuptools to install the pympanim module"""
import setuptools
import os

setuptools.setup(
    name='pympanim',
    version='0.0.6',
    author='Timothy Moore',
    author_email='mtimothy984@gmail.com',
    description='Multiprocessing-friendly animations',
    license='CC0',
    keywords='pympanim animations video mp4',
    url='https://github.com/tjstretchalot/pympanim',
    packages=['pympanim'],
    long_description=open(
        os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    long_description_content_type='text/markdown',
    install_requires=['pyzmq', 'pytypeutils', 'Pillow'],
    classifiers=(
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication',
        'Topic :: Utilities'),
    python_requires='>=3.6',
)
