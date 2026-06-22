# Building diff-surfel-rasterization

## Important: Always use the rebuild script!

After making changes to CUDA code, **ALWAYS** use:

```bash
./rebuild.sh
```

or with conda:

```bash
conda run -n nest_splatting ./rebuild.sh
```

## Why?

Python loads modules from `site-packages` by default, not from the local build directory. The rebuild script ensures:

1. Compiles the CUDA extension
2. **Copies the compiled .so to site-packages** (so Python uses it)
3. Copies Python files to site-packages

## DO NOT use these (they won't update site-packages):

❌ `python setup.py build_ext --inplace`  
❌ `python setup.py build`

## Alternative: Editable install

You can also do a proper editable install once:

```bash
pip install -e .
```

Then rebuilds will automatically update site-packages.

