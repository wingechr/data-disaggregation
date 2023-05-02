from setuptools import setup

with open("README.md", encoding="utf-8") as file:
    long_description_md = file.read()

if __name__ == "__main__":
    setup(
        packages=["data_disaggregation"],
        name="data-disaggregation",
        install_requires=["pandas"],
        keywords=[],
        description="",
        long_description=long_description_md,
        long_description_content_type="text/markdown",
        version="0.9.2",
        author="Christian Winger",
        platforms=["any"],
        license="MIT",
        project_urls={
            "Documentation": "https://wingechr.github.io/data-disaggregation",
            "Source": "https://github.com/wingechr/data-disaggregation",
        },
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
    )
