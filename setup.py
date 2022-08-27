import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

PROJECT_NAME="XGB_binary_classifier"
USER_NAME="Basavaraj100"

setuptools.setup(
    name=f"{PROJECT_NAME}-{USER_NAME}",
    version="0.0.2",
    author=PROJECT_NAME,
    author_email="benkibijali@gmail.com",
    description="Fit XGB binary classifier for given train data and generate save Gain, KS for train and test data and also generate sweetviz comparison between the train and tests",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{USER_NAME}/{PROJECT_NAME}",
    project_urls={ 
        "Bug Tracker": f"https://github.com/{USER_NAME}/{PROJECT_NAME}/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7"
    install_requires=['pandas','numpy','matplotlib','seaborn','sklearn','xgboost','sweetviz']

    ,
)