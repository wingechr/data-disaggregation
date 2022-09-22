from setuptools import find_packages, setup

if __name__ == "__main__":

    # long description from readme
    with open("README.md", encoding="utf-8") as file:
        long_description = file.read()

    setup(
        packages=find_packages(),
        name="data-disaggregation",
        install_requires=["numpy"],
        keywords=[],
        description="Data (Dis-)aggregation tool",
        long_description=long_description,
        # text/markdown or text/x-rst or text/plain
        long_description_content_type="text/markdown",
        version="0.5.0",
        author="Christian Winger",
        author_email="c.winger@oeko.de",
        url="https://github.com/wingechr/data-disaggregation",
        platforms=["any"],
        license="MIT",
        project_urls={
            "Bug Tracker": "https://github.com/wingechr/data-disaggregation",
            "Documentation": "https://data-disaggregation.readthedocs.io",
        },
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
    )
