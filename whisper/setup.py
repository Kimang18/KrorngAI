# Author: KrorngAI org
# Date: December 2025


import sys
from pathlib import Path
import os
import shutil

from setuptools import setup, find_namespace_packages

package_dir = Path(__file__).parent / 'neo_whisper'

# Path to common script outside the package root
for script in ["nn_utils.py", "common.py"]:
    common_script_src = os.path.join("..", "common_utils", script)
    common_script_dest = os.path.join(str(package_dir), script)
    if os.path.exists(common_script_src):
        shutil.copy(common_script_src, common_script_dest)

sys.path.append(str(package_dir))

from _version import __version__

try:
    setup(
        name='neo-whisper',
        version=__version__,
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
            # 'requests>=2.25.1',
            # GitHub dependency using PEP 508 URL format
            # 'openai-whisper @ git+https://github.com/openai/whisper.git'
        ]
    )
finally:
    for script in ["nn_utils.py", "common.py"]:
        common_script_dest = os.path.join(str(package_dir), script)
        if os.path.exists(common_script_dest):
            os.remove(common_script_dest)
