# nlp_project
# Make sure you have upgraded pip
Windows
```
py -m pip install --upgrade pip 
```

# Project Structure
```
reader/
│
├── reader/
│   ├── config.txt
│   ├── classifier.py
│   ├── __init__.py
│   ├── __main__.py
│   └── token.py
│
├── tests/
│
├── MANIFEST.in
├── README.md
└── setup.py
touch LICENSE
touch nlp_project.toml
touch setup.cfg
mkdir src/nlp_project
touch src/nlp_project/__init__.py
touch src/nlp_project/main.py
mkdir tests
```

# nlp_project.toml
[build-system]
requires = [
    "setuptools>=42",
    "wheel"
]
build-backend = "setuptools.build_meta"

# setup
```
[metadata]
name = example-pkg-YOUR-USERNAME-HERE
version = 0.0.1
author = Example Author
author_email = author@example.com
description = A small example package
long_description = file: README.md
long_description_content_type = text/markdown
url = https://github.com/pypa/sampleproject
project_urls =
    Bug Tracker = https://github.com/pypa/sampleproject/issues
classifiers =
    Programming Language :: Python :: 3
    License :: OSI Approved :: MIT License
    Operating System :: OS Independent

[options]
package_dir =
    = src
packages = find:
python_requires = >=3.9

[options.packages.find]
where = src
```

