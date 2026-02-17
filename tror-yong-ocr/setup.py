# Author: KrorngAI org
# Date: February 2026


import sys
from pathlib import Path

from setuptools import setup, find_namespace_packages

package_dir = Path(__file__).parent / 'tror_yong_ocr'

sys.path.append(str(package_dir))

from _version import __version__


setup(
    name='tror-yong-ocr',
    version=__version__,
    description='Optical Character Recognition Model',
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
        # 'requests>=2.25.1',
        # GitHub dependency using PEP 508 URL format
        # 'openai-whisper @ git+https://github.com/openai/whisper.git'
    ]
)
