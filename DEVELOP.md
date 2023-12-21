# Develop

## Test

```bash
python -m unittest
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
