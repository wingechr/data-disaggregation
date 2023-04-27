from setuptools import setup

if __name__ == "__main__":
    setup(
        packages=["data_disaggregation"],
        name="data-disaggregation",
        install_requires=["pandas"],
        keywords=[],
        description="",
        long_description="",
        long_description_content_type="text/markdown",
        version="0.9.0",
        author="Christian Winger",
        platforms=["any"],
        license="MIT",
        project_urls={
            "Documentation": "https://data-disaggregation.readthedocs.io",
        },
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
    )
