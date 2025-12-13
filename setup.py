from setuptools import setup, find_packages


setup(
    name='neo_whisper',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        # add dependencies here
        'requests>=2.25.1',
        # GitHub dependency using PEP 508 URL format
        'openai-whisper @ git+https://github.com/openai/whisper.git'
    ]
)
