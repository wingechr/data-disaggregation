import re

from setuptools import setup

# short description package docstring
import data_disaggregation as pkg

if __name__ == "__main__":

    # long description from readme
    with open("README.md", encoding="utf-8") as file:
        long_description = file.read()

    short_description = re.sub(r"\s+", " ", pkg.__doc__).strip()

    setup(
        packages=["data_disaggregation"],
        name="data-disaggregation",
        install_requires=["numpy"],
        keywords=[],
        description=short_description,
        long_description=long_description,
        # text/markdown or text/x-rst or text/plain
        long_description_content_type="text/markdown",
        version=pkg.__version__,
        author=pkg.__author__,
        author_email=pkg.__email__,
        maintainer=pkg.__author__,
        maintainer_email=pkg.__email__,
        url=pkg.__url__,
        download_url=pkg.__url__,
        platforms=["any"],
        license=pkg.__copyright__,
        project_urls={
            "Bug Tracker": pkg.__url__,
            "Documentation": "https://data-disaggregation.readthedocs.io",
        },
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",  # noqa: E501
            "Operating System :: OS Independent",
        ],
        entry_points={
            "console_scripts": [
                # "cmd = PACKAGE_NAME.scripts.NAME:main"
            ]
        },
        package_data={
            # "package.module": [file_patterns]
        },
    )
