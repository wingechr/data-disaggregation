# Develop

## Test

```bash
# quick test
python -m unittest

# full test
tox
```

## Documentation

```bash
mkdocs build
```

## Build & Publish

```bash
rm -rf dist
python setup.py sdist
twine upload dist/*.tar.gz
```
