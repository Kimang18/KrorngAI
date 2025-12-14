# Author: KrorngAI org
# Date: December 2025


import sys
from pathlib import Path

from setuptools import setup, find_namespace_packages

package_dir = Path(__file__).parent / 'neo_whisper'

sys.path.append(str(package_dir))

setup(
    name='neo-whisper',
    version='0.0.1',
    description='Improve Whisper with RoPE and latest tokenizers of OpenAI',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    author_email="kimang.khun@polytechnique.org",
    author="KHUN Kimang",
    url="https://github.com/kimang18/KrorngAI",
    packages=find_namespace_packages(),
    python_requires=">=3.8",
    install_requires=[
        # add dependencies here
        'requests>=2.25.1',
        # GitHub dependency using PEP 508 URL format
        'openai-whisper @ git+https://github.com/openai/whisper.git'
    ]
)
