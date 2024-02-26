from setuptools import find_packages, setup

setup(
    name="cremp",
    version="1.0.0",
    url="https://github.com/Genentech/cremp",
    author="Colin Grambow",
    author_email="grambow.colin@gene.com",
    packages=find_packages(),
    install_requires=[],
    scripts=["cremp/run_crest.py"],
)
