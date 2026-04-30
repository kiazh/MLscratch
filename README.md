# MLscratch

I wanted to build this because I was interested in machine learning and wanted to understand how it works from the ground up — no libraries, just C.

Inspired by and based on the tutorial by [Magicalbat](https://www.youtube.com/@Magicalbat): [Machine Learning in C](https://www.youtube.com/watch?v=hL_n_GljC0I)

---

## How It Works

## Architecture

## Training

## Results

## Building and Running

### macOS / Linux

```bash
gcc src/main.c -o src/main -lm && ./src/main
```

### Windows (MinGW / MSYS2)

```bat
gcc src\main.c -o src\main.exe -lm -lbcrypt && src\main.exe
```

> Run from the `MLscratch` directory. Requires `git lfs pull` after cloning to download the MNIST data files.

## Dependencies

- C compiler (gcc / clang / MinGW)
- Git LFS (for data files)
