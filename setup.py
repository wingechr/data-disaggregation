from setuptools import setup

from docs import conf

if __name__ == "__main__":

    with open("README.md", encoding="utf-8") as file:
        long_description = file.read()

    setup(
        packages=["data_disaggregation"],
        keywords=[],
        install_requires=[],
        extras_require={"dev": []},
        name="data-disaggregation",
        description=conf.description,
        long_description=long_description,
        # text/markdown or text/x-rst or text/plain
        long_description_content_type="text/markdown",
        version=conf.version,
        author=conf.author,
        author_email=conf.email,
        maintainer=conf.author,
        maintainer_email=conf.email,
        url=conf.urls["code"],
        download_url=conf.urls["code"],
        platforms=["any"],
        license=conf.copyright,
        project_urls={
            "Bug Tracker": conf.urls["code"],
        },
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
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
