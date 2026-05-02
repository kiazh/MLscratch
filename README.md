# MLscratch

A feedforward neural network written in C from scratch — no external libraries, just the C standard library and `libm`. Trains on MNIST and achieves ~97% test accuracy.

Inspired by [Magicalbat's Machine Learning in C](https://www.youtube.com/watch?v=hL_n_GljC0I).

---

## How It Works

Four layers of hand-rolled infrastructure sit under the model:

| Layer | File | What it does |
|---|---|---|
| Types & macros | `base.h` | Sized integer aliases (`u32`, `f32`, etc.) and size helpers (`KiB`, `MiB`, `GiB`) |
| Memory | `arena.h/c` | Virtual-memory arena allocator — reserve once, commit on demand, scratch arenas for temporaries |
| PRNG | `psgrdn.h/c` | PCG32 generator seeded from OS entropy; uniform and normal (Box-Muller) distributions |
| Matrix | `main.c` | Row-major `float` matrices with all four transpose variants of matmul |

On top of those sits a **computation graph** (`model_var` / `model_program`). Each node records its op and input pointers. `model_prog_create` topologically sorts the graph once at compile time; `model_prog_compute` runs the forward pass and `model_prog_compute_grads` runs the backward pass over the same sorted list in reverse.

## Architecture

Three fully-connected layers trained on 28×28 greyscale MNIST images (flattened to 784 floats):

```
Input (784)
  → Linear(784→16) + Bias + ReLU       → a0
  → Linear(16→16)  + Bias + ReLU + a0  → a1   ← residual skip from layer 0
  → Linear(16→10)  + Bias + Softmax    → output (10 class probabilities)
```

Weights are initialised with **Xavier uniform** (`±√(6 / (fan_in + fan_out))`). Biases start at zero.

Loss: **cross-entropy** between the one-hot label and the softmax output.

## Training

Mini-batch **SGD** with a shuffled training order each epoch.

| Hyperparameter | Value |
|---|---|
| Epochs | 3 |
| Batch size | 50 |
| Learning rate | 0.01 |

Training prints live `Epoch / Batch / Cost` progress and a full test-set accuracy report after each epoch.

## Results

On MNIST (60 000 train / 10 000 test):

```
Epoch  1 — ~95% test accuracy
Epoch  2 — ~96% test accuracy
Epoch  3 — ~97% test accuracy
```

Exact numbers vary per run due to random weight init and batch ordering.

## Building and Running

> Requires [Git LFS](https://git-lfs.github.com/) — run `git lfs pull` after cloning to fetch the MNIST `.npy` data files.

### macOS / Linux (Make)

```bash
make
./src/main
```

### macOS / Linux (manual)

```bash
clang -Wall -Wextra -O2 src/main.c -o src/main -lm
./src/main
```

### Windows (MinGW / MSYS2)

```bat
gcc src\main.c -o src\main.exe -lm -lbcrypt
src\main.exe
```

Run from the repository root so the binary can find the `data/` directory.

## Project Structure

```
MLscratch/
├── src/
│   ├── base.h       — sized type aliases, size macros
│   ├── arena.h/c    — virtual-memory arena allocator
│   ├── psgrdn.h/c   — PCG32 PRNG (uniform + normal)
│   └── main.c       — matrix ops, computation graph, model, training loop
├── data/            — MNIST .npy files (fetched via Git LFS)
└── Makefile
```

## Dependencies

- C compiler (clang / gcc / MinGW)
- Git LFS (for MNIST data files)

## Resources

- [Magicalbat — Machine Learning in C](https://www.youtube.com/watch?v=hL_n_GljC0I)
- [3Blue1Brown — Neural Networks series](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Adam Dhala — Backpropagation](https://www.youtube.com/watch?v=Ixl3nykKG9M)
- [Artem Kirsanov — Neural networks](https://www.youtube.com/watch?v=SmZmBKc7Lrs&list=PLgtmMKe4spCPsxyMpg-sxf3EcbsFYlzPK)
- [Nielsen — Neural Networks and Deep Learning](https://github.com/mnielsen/neural-networks-and-deep-learning)
