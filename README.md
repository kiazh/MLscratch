# MLscratch

A neural network library written in C from scratch. There are no external dependencies beyond the C standard library and `<math.h>`. The project implements a small automatic differentiation engine and trains a three-layer multilayer perceptron on the MNIST handwritten digit dataset, achieving approximately 98% test accuracy.

The implementation is based on the tutorial by [Magicalbat](https://www.youtube.com/@Magicalbat): [Machine Learning in C](https://www.youtube.com/watch?v=hL_n_GljC0I).

---

## Table of Contents

1. [Background: Neural Networks](#background-neural-networks)
2. [The Calculus: Gradient Derivations](#the-calculus-gradient-derivations)
3. [Code Architecture](#code-architecture)
4. [The MNIST Model](#the-mnist-model)
5. [The Training Procedure](#the-training-procedure)
6. [Implementation Notes](#implementation-notes)
7. [Results](#results)
8. [Building and Running](#building-and-running)
9. [Resources Used](#resources-used)

---

## Background: Neural Networks

### The neuron

A neuron is the smallest computational unit of a neural network. It takes a vector of inputs, multiplies each by a *weight*, adds the products together along with a *bias* term, and passes the result through a nonlinear *activation function*:

$$y = f(w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b)$$

The weights $w_i$ control the relative influence of each input. The bias $b$ shifts the result. The function $f$ introduces nonlinearity.

A *layer* is a collection of neurons that share the same input vector. Its forward computation can be written as a single matrix multiplication:

$$\mathbf{z} = W \mathbf{x} + \mathbf{b}$$

where $W$ contains one row per neuron and one column per input. This equation is the foundation of all densely connected neural networks. In this codebase it appears as a call to `mat_mul` followed by `mat_add`.

### Why activation functions exist

Composing two linear layers without an intervening nonlinearity yields another linear function: $W_2(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (W_2 W_1)\mathbf{x} + (W_2 \mathbf{b}_1 + \mathbf{b}_2)$. A network of arbitrary depth would therefore collapse to a single layer. Nonlinearities are required to make depth meaningful.

This codebase uses two activation functions.

**ReLU** (Rectified Linear Unit) is used in hidden layers:

$$\text{ReLU}(x) = \max(0, x)$$

Its derivative is 1 for positive inputs and 0 otherwise. It is fast to compute, does not saturate for large inputs, and propagates gradients cleanly along active paths.

**Softmax** is used in the output layer for classification. It maps a vector of real-valued *logits* to a probability distribution:

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

The output is positive, sums to 1, and preserves the relative ordering of the logits.

### The forward pass

Given an input vector (in this case the 784 normalized pixel values of a 28 by 28 MNIST image), the forward pass chains the operations above:

$$\mathbf{a}_0 = \text{ReLU}(W_0 \mathbf{x} + \mathbf{b}_0)$$

$$\mathbf{a}_1 = \mathbf{a}_0 + \text{ReLU}(W_1 \mathbf{a}_0 + \mathbf{b}_1)$$

$$\hat{\mathbf{y}} = \text{softmax}(W_2 \mathbf{a}_1 + \mathbf{b}_2)$$

The output $\hat{\mathbf{y}}$ is a vector of ten probabilities. The network's prediction is the index of the largest component.

### The loss function

Training requires a scalar measure of error. For classification with softmax outputs, the standard choice is **cross-entropy loss**:

$$L = -\sum_i y_i \log(\hat{y}_i)$$

where $\mathbf{y}$ is the true label encoded as a one-hot vector and $\hat{\mathbf{y}}$ is the predicted distribution. Because $\mathbf{y}$ is one-hot, only one term contributes, reducing the loss to $-\log(\hat{y}_\text{correct})$. The loss approaches zero when the network assigns high probability to the correct class and grows without bound as that probability approaches zero.

### Gradient descent

Each weight in the network is a parameter. The loss is a differentiable function of every parameter. The **gradient** $\partial L / \partial w$ specifies the direction of steepest increase in loss with respect to $w$. Moving in the opposite direction decreases the loss:

$$w \leftarrow w - \eta \frac{\partial L}{\partial w}$$

The scalar $\eta$ is the **learning rate** (this codebase uses 0.01). Repeating this update over many examples drives the parameters toward values that minimize the loss.

### Backpropagation

A direct evaluation of $\partial L / \partial w$ for every parameter would be intractable. **Backpropagation** computes all gradients in a single backward pass by applying the chain rule. If $L$ depends on $a$ and $a$ depends on $w$, then

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial w}$$

A neural network is a composition of differentiable operations. By recording these operations during the forward pass and traversing them in reverse, multiplying local derivatives at each step, all gradients are computed in time proportional to the forward pass. This requires knowing the local derivative rule for each operation. These rules are derived in the next section.

### Stochastic gradient descent

Computing the gradient over the entire training set before each update is unnecessary and slow. **Mini-batch stochastic gradient descent** instead averages gradients over a small random subset of examples (the *batch*) and updates after each batch. This codebase uses a batch size of 50. The randomness across batches improves convergence and acts as implicit regularization.

A full pass through the training set is one **epoch**. The training procedure shuffles the data between epochs. This codebase trains for three epochs.

---

## The Calculus: Gradient Derivations

Each operation in the autograd engine has an associated backward rule. The derivations follow.

### Addition: $C = A + B$

The Jacobian of addition with respect to either input is the identity. Therefore:

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C}, \qquad \frac{\partial L}{\partial B} = \frac{\partial L}{\partial C}$$

The gradient passes through unchanged to both operands. This property is what makes residual connections effective: the gradient receives an unobstructed path backward through the addition node.

### Subtraction: $C = A - B$

The same as addition for the first operand, with a sign flip for the second:

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C}, \qquad \frac{\partial L}{\partial B} = -\frac{\partial L}{\partial C}$$

### Matrix multiplication: $C = AB$

Each entry $C_{ij} = \sum_k A_{ik} B_{kj}$. The partial derivative $\partial C_{ij} / \partial A_{mn}$ equals $B_{nj}$ when $m = i$ and zero otherwise. Applying the chain rule:

$$\frac{\partial L}{\partial A_{mn}} = \sum_{i,j} \frac{\partial L}{\partial C_{ij}} \frac{\partial C_{ij}}{\partial A_{mn}} = \sum_j \frac{\partial L}{\partial C_{mj}} B_{nj} = \left(\frac{\partial L}{\partial C} B^T\right)_{mn}$$

Similarly for $B$:

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} B^T, \qquad \frac{\partial L}{\partial B} = A^T \frac{\partial L}{\partial C}$$

The implementation uses `mat_mul` with the transpose flags appropriately set, avoiding any explicit transpose:

```c
case MV_OP_MATMUL: {
    if (a requires grad) mat_mul(a->grad, cur->grad, b->val, 0, 0, 1); // dL/dC · B^T
    if (b requires grad) mat_mul(b->grad, a->val, cur->grad, 0, 1, 0); //   A^T · dL/dC
}
```

The flag arguments are `(zero_out, transpose_a, transpose_b)`. The `zero_out` flag is false because gradients accumulate: a single parameter may receive contributions from multiple paths in the graph and from multiple examples in a batch.

### ReLU: $y_i = \max(0, x_i)$

The derivative is 1 where $x_i$ is strictly positive and 0 elsewhere. The point $x_i = 0$ is a kink; the convention adopted here returns 0:

$$\frac{\partial L}{\partial x_i} = \begin{cases} \partial L / \partial y_i & \text{if } x_i > 0 \\ 0 & \text{otherwise} \end{cases}$$

### Softmax: $s_i = e^{z_i} / \sum_j e^{z_j}$

Softmax is a vector-valued function. Its derivative is a Jacobian matrix. The diagonal terms are $\partial s_i / \partial z_i = s_i (1 - s_i)$. The off diagonal terms are $\partial s_i / \partial z_j = -s_i s_j$. These combine using the Kronecker delta:

$$\frac{\partial s_i}{\partial z_j} = s_i (\delta_{ij} - s_j)$$

The chain rule for a vector-valued intermediate is

$$\frac{\partial L}{\partial z_j} = \sum_i \frac{\partial L}{\partial s_i} \frac{\partial s_i}{\partial z_j}$$

which is the matrix vector product $J \cdot (\partial L / \partial \mathbf{s})$. The Jacobian is symmetric, so transposition is unnecessary. The implementation constructs the full $N \times N$ Jacobian and multiplies it against the upstream gradient:

```c
for (u32 i = 0; i < size; i++) {
    for (u32 j = 0; j < size; j++) {
        jacobian->data[j + i*size] = softmax_out->data[i] * ((i == j) - softmax_out->data[j]);
    }
}
mat_mul(out, jacobian, grad, 0, 0, 0);
```

This is correct but expensive: $O(N^2)$ memory and $O(N^3)$ time per backward pass. For $N = 10$ the cost is negligible.

### Cross-entropy: $L_i = -p_i \log(q_i)$ (per element)

Here $p$ denotes the true distribution and $q$ the predicted distribution. The partial derivatives are

$$\frac{\partial L_i}{\partial q_i} = -\frac{p_i}{q_i}, \qquad \frac{\partial L_i}{\partial p_i} = -\log(q_i)$$

The implementation includes an additive epsilon of $10^{-7}$ to prevent division by zero and the logarithm of zero:

```c
p_grad[i] += scale * -logf(q[i] + 1e-7f);
q_grad[i] += scale * -p[i] / (q[i] + 1e-7f);
```

The scale variable is the upstream gradient $\partial L / \partial L_i$. During training it is initialized to 1 by `mat_fill(cost->grad, 1.0f)` at the top of `model_prog_compute_grads`.

### The combined softmax cross-entropy gradient

When the softmax and cross-entropy gradients are composed analytically, substantial cancellation occurs. For a one-hot target $\mathbf{y}$ and softmax output $\hat{\mathbf{y}}$, the gradient with respect to the pre softmax logits reduces to:

$$\frac{\partial L}{\partial \mathbf{z}} = \hat{\mathbf{y}} - \mathbf{y}$$

Most production frameworks fuse the two operations and use this closed form for both speed and numerical stability. This codebase intentionally does not. Implementing the operations separately preserves graph generality: substituting a different output activation or loss function requires no changes elsewhere.

---

## Code Architecture

The codebase implements a small automatic differentiation engine. A computation graph is constructed as data, compiled into a topologically ordered execution program, and evaluated forward and backward generically.

### The arena allocator

All memory allocation passes through arenas (see `arena.c` and `arena.h`). An arena reserves a contiguous block of address space (1 GiB by default) and commits physical pages as allocations advance. Reservation uses `mmap` with `PROT_NONE` on Linux and macOS and `VirtualAlloc` with `MEM_RESERVE` on Windows. Commits use `mprotect` and `VirtualAlloc` with `MEM_COMMIT` respectively. Allocations are bumped onto the end of the used region; deallocation occurs only when the arena is destroyed or rolled back to a saved position. This eliminates fragmentation and prevents leaks.

Two thread local **scratch arenas** provide ephemeral workspace for operations such as `mat_softmax_add_grad`, which constructs a temporary Jacobian matrix:

```c
mem_arena_temp scratch = arena_scratch_get(NULL, 0);
matrix* jacobian = mat_create(scratch.arena, size, size);
// ... use jacobian ...
arena_scratch_release(scratch);
```

A scratch arena's position pointer is rolled back on release, freeing all temporary allocations atomically. The conflicts argument supports caller specified exclusion to prevent reentrant use of the same arena.

### The matrix type

All numerical data is stored in a `matrix` structure containing dimensions and a row major float buffer:

```c
typedef struct {
    uint32_t rows, cols;
    float *data;
} matrix;
```

The matrix module provides element wise operations (`mat_add`, `mat_sub`, `mat_scale`, `mat_clear`, `mat_fill`, `mat_fill_rand`), reductions (`mat_sum`, `mat_argmax`), the activation primitives (`mat_relu`, `mat_softmax`), the loss primitive (`mat_cross_entropy_loss`), and their associated gradient routines.

The function `mat_mul` accepts flags `(zero_out, transpose_a, transpose_b)` and dispatches to one of four loop variants for the four transpose combinations. This avoids materializing transposes during backprop. The inner loops use `i, k, j` ordering rather than `i, j, k` to improve cache locality on row major data.

### The autograd graph

Every quantity in the network is represented as a `model_var`:

```c
typedef struct model_var {
    u32 index;
    u32 flags;            // INPUT, OUTPUT, PARAMETER, REQUIRES_GRAD, COST, ...
    matrix* val;
    matrix* grad;
    model_var_op op;      // ADD, SUB, MATMUL, RELU, SOFTMAX, CROSS_ENTROPY
    struct model_var* inputs[2];
} model_var;
```

A call to `mv_matmul(arena, model, W0, input, 0)` does not perform any computation. It allocates a new `model_var` whose op code is `MV_OP_MATMUL` and whose input pointers reference `W0` and `input`. Constructing a network is therefore a series of structural assignments that mirror the mathematical specification:

```c
model_var* z0_a = mv_matmul(arena, model, W0, input, 0);   // z0 = W0 · x
model_var* z0_b = mv_add(arena, model, z0_a, b0, 0);       // z0 += b0
model_var* a0   = mv_relu(arena, model, z0_b, 0);          // a0 = ReLU(z0)
```

Op codes are partitioned by arity using sentinel constants (`_MV_OP_UNARY_START`, `_MV_OP_BINARY_START`). The macro `MV_NUM_INPUTS(op)` returns 0, 1, or 2 based on which range an op falls in. The flag `MV_FLAG_REQUIRES_GRAD` propagates automatically: any output of an operation involving a requires-grad input is itself flagged.

### Compilation

The function `model_compile` traverses the graph from the output node backward via iterative depth first search (see `model_prog_create`). It produces a flat array of nodes in topological order, in which every node appears after its dependencies. This permits forward evaluation by left-to-right iteration and backward evaluation by right-to-left iteration without recursion.

Two programs are compiled. The `forward_prog` evaluates the network output and is used during inference. The `cost_prog` evaluates the loss and is used during training. The DFS uses a visit twice strategy: each node is pushed onto a stack, and on its second pop (after its children have completed) it is appended to the output array.

### Forward evaluation

`model_prog_compute` iterates the program array forward. For each node, it dispatches on the op code and invokes the corresponding matrix routine. After completion, every `val` field in the program is populated, including the loss.

### Backward evaluation

`model_prog_compute_grads` proceeds in three phases. First, it iterates forward and clears the gradient buffer of every requires-grad node that is not a parameter. Parameter gradients are deliberately preserved across calls so that gradients accumulate across the examples in a mini-batch. Second, it seeds the gradient of the loss node by filling its gradient buffer with 1.0. Third, it iterates the program array backward and applies the local gradient rule for each op, accumulating contributions into the input nodes' gradient buffers.

```c
case MV_OP_MATMUL:
    if (a needs grad) mat_mul(a->grad, cur->grad, b->val, 0, 0, 1);
    if (b needs grad) mat_mul(b->grad, a->val, cur->grad, 0, 1, 0);

case MV_OP_ADD:
    mat_add(a->grad, a->grad, cur->grad);
    mat_add(b->grad, b->grad, cur->grad);

case MV_OP_SUB:
    mat_add(a->grad, a->grad, cur->grad);
    mat_sub(b->grad, b->grad, cur->grad);

case MV_OP_RELU:
    mat_relu_add_grad(a->grad, a->val, cur->grad);

case MV_OP_SOFTMAX:
    mat_softmax_add_grad(a->grad, cur->val, cur->grad);

case MV_OP_CROSS_ENTROPY:
    mat_cross_entropy_add_grad(p->grad, q->grad, p->val, q->val, cur->grad);
```

When the loop terminates, every parameter gradient buffer holds the accumulated gradient ready for the update step.

---

## The MNIST Model

The function `create_mnist_model` constructs the network:

```
input          : 784 × 1     (flattened 28 × 28 image, normalized to [0,1])

W0 (16 × 784) ──┐
                ├─→ matmul → +b0 → ReLU ──→ a0 (16 × 1)
                │
W1 (16 × 16)  ──┐
                ├─→ matmul → +b1 → ReLU ──→ z1_c
                                              │
                                  a0 ────────►├─ add (residual) ──→ a1 (16 × 1)
                                              │
W2 (10 × 16)  ──┐
                ├─→ matmul → +b2 → softmax ──→ output (10 × 1)
                │
y (10 × 1) ─────┴─→ cross_entropy(y, output) ──→ cost
```

Total trainable parameter count: $W_0$ contributes $16 \times 784 = 12{,}544$; $b_0$ contributes 16; $W_1$ contributes 256; $b_1$ contributes 16; $W_2$ contributes 160; $b_2$ contributes 10. The total is **13,002 parameters**.

### The residual skip connection

The line `mv_add(arena, model, a0, z1_c, 0)` constructs a residual block by adding the first hidden layer's output directly onto the second:

$$\mathbf{a}_1 = \mathbf{a}_0 + \text{ReLU}(W_1 \mathbf{a}_0 + \mathbf{b}_1)$$

Because addition propagates the gradient unchanged to both operands, the gradient at $\mathbf{a}_1$ reaches $\mathbf{a}_0$ through two paths: the deep path through the matmul and ReLU, and the direct skip. Even when the deep path attenuates the gradient (for example when ReLU zeros negative pre-activations), the skip path preserves signal. This mitigates vanishing gradients and is the core idea behind ResNet.

### Weight initialization (Xavier/Glorot uniform)

Weights are sampled from a uniform distribution:

$$W \sim \mathcal{U}\left[-\sqrt{\frac{6}{n_\text{in} + n_\text{out}}}, \;+\sqrt{\frac{6}{n_\text{in} + n_\text{out}}}\right]$$

The constant 6 is derived. The variance of a uniform distribution on $[-a, a]$ is $a^2 / 3$. Setting $a = \sqrt{6 / (n_\text{in} + n_\text{out})}$ produces $\text{Var}(W) = 2 / (n_\text{in} + n_\text{out})$, the value Glorot and Bengio derived to maintain approximately constant activation variance across layers at initialization.

```c
f32 bound0 = sqrtf(6.0f / (784 + 16));
f32 bound1 = sqrtf(6.0f / (16 + 16));
f32 bound2 = sqrtf(6.0f / (16 + 10));
mat_fill_rand(W0->val, -bound0, bound0);
mat_fill_rand(W1->val, -bound1, bound1);
mat_fill_rand(W2->val, -bound2, bound2);
```

Biases are initialized to zero, which is the default state of arena allocated memory.

### The pseudorandom number generator

The file `psgrdn.c` implements a Permuted Congruential Generator (PCG), specifically the XSH-RR variant:

```c
rng->state = oldstate * 6364136223846793005ULL + rng->inc;
uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
uint32_t rot = oldstate >> 59u;
return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
```

A standard 64 bit linear congruential update is followed by an output permutation that combines an xorshift fold and a state dependent rotation. PCG generators pass the TestU01 BigCrush statistical test suite, which standard LCGs fail. The seed is drawn from operating system entropy: `getentropy` on Linux and macOS, `BCryptGenRandom` on Windows.

The module also includes a Box-Muller implementation (`prng_rand_norm`) for sampling from a standard normal distribution, with caching of the second output via the `prev_norm` field. This routine is unused in the current MNIST initialization path.

### Data loading

The MNIST dataset is loaded from `.npy` files prepared by `scripts/nist.py`. The script uses Keras to obtain MNIST and writes float32 arrays normalized to $[0, 1]$. The C loader (`mat_load`) parses the numpy header (magic bytes, version, header length field), seeks past it, and reads the raw float buffer directly into a matrix. Labels arrive as integers in $\{0, ..., 9\}$ and are immediately encoded as one-hot vectors.

The function `draw_mnist_digit` renders an image to the terminal using ANSI 256 color background escape codes, mapping pixel intensity to the 24 step grayscale ramp at color indices 232 through 255.

---

## The Training Procedure

The function `model_train` implements the training loop. For each of the three epochs, the training indices are first shuffled. The shuffle iterates 60,000 times, picking two random indices $a$ and $b$ and swapping `training_order[a]` with `training_order[b]`. The training set is then divided into 1,200 mini-batches of 50 examples.

For each mini-batch the procedure is as follows. First, parameter gradient buffers are zeroed by walking the cost program and calling `mat_clear` on each parameter's gradient. Non-parameter gradients are zeroed inside `model_prog_compute_grads`. Second, the batch is processed one example at a time: the input image is copied into `model->input->val`, the one-hot label is copied into `model->desired_output->val`, and the forward and backward programs are evaluated. Because all gradient operations accumulate by addition rather than overwrite, parameter gradient buffers hold the sum of all 50 per-example gradients after the batch completes. Third, parameters are updated. The accumulated gradient is scaled by $\eta / B$ (combining the learning rate and batch averaging in one operation) and subtracted from the parameter:

```c
mat_scale(cur->grad, learning_rate / batch_size);
mat_sub(cur->val, cur->val, cur->grad);
```

After each epoch the entire test set is evaluated by running the cost program over all 10,000 test examples. The number of correct predictions is accumulated by comparing `mat_argmax(output)` against `mat_argmax(desired_output)`. The accuracy and average loss are reported.

The implementation processes one example at a time; it does not batch examples into a single matrix. A production implementation would stack the 50 examples into a $784 \times 50$ input matrix and execute one batched matmul per layer, achieving substantially higher throughput due to better cache and floating point unit utilization. The present design prioritizes simplicity of the autograd graph over throughput.

---

## Implementation Notes

The following are deliberate design choices and characteristics of the implementation, documented for transparency.

**Softmax is not numerically stabilized.** `mat_softmax` calls `expf(in[i])` directly. For sufficiently large input values this overflows to infinity. The numerically stable formulation $e^{z_i - \max(z)} / \sum_j e^{z_j - \max(z)}$ is mathematically equivalent. At the scale of this network and with Xavier initialization, overflow is not observed in practice.

**The shuffle is statistically biased.** A swap of two random indices repeated $N$ times does not produce a uniform distribution over permutations. The number of distinct outcomes is $N^{2N}$, whereas the number of permutations is $N!$, and the former is generally not divisible by the latter. The Fisher-Yates algorithm, which performs a single pass swapping each position with a uniformly chosen later position, is unbiased. For mini-batch SGD any reasonable mixing suffices and the bias has no observable training effect.

**The fused softmax cross-entropy gradient is not used.** As noted in the calculus section, the closed form $\hat{\mathbf{y}} - \mathbf{y}$ would replace an $O(N^2)$ Jacobian construction and an $N \times N$ matrix multiplication. The two operations are kept separate to preserve graph generality. For $N = 10$ the cost difference is negligible.

**The cost is a vector, not a scalar.** The `cost` node holds a length 10 vector of per-element cross-entropy contributions. The scalar loss is the sum of these components, computed by `mat_sum(model->cost->val)`. Because cross-entropy with a one-hot target has only one nonzero term, this sum equals $-\log(\hat{y}_\text{correct})$. The backward pass seeds the loss node's gradient with the all ones vector, which is the gradient of the elementwise sum with respect to each component.

**Parameter gradients accumulate across `model_prog_compute_grads` calls.** Non-parameter requires-grad gradients are cleared at the start of each backward pass; parameter gradients are not. This is what enables mini-batch gradient summation. The training loop is responsible for clearing parameter gradients between batches.

---

## Results

After three epochs of training (approximately one minute on a modern laptop), the network achieves approximately **98% test accuracy** on MNIST. With 13,002 parameters, no convolutional structure, no dropout, no momentum, and no adaptive optimizer, this result demonstrates that the autograd engine and gradient implementations are correct.

Before training, the network output is approximately uniform across the ten classes. After training, the network places the majority of its probability mass on the correct digit.

---

## Building and Running

### macOS / Linux

```bash
gcc src/main.c -o src/main -lm && ./src/main
```

### Windows (MinGW / MSYS2)

```bash
gcc src\main.c -o src\main.exe -lm -lbcrypt && src\main.exe
```

The program must be run from the `MLscratch` directory. Cloning the repository requires `git lfs pull` to obtain the MNIST data files.

### Dependencies

A C compiler (gcc, clang, or MinGW). Git LFS for the data files. Python with `keras` and `numpy` is required only to regenerate the `.npy` files via `scripts/nist.py`.

---

## Resources Used

[3Blue1Brown Series](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=2)

[Adam Dhala](https://www.youtube.com/watch?v=Ixl3nykKG9M)

[Artem Kirsanov](https://www.youtube.com/watch?v=SmZmBKc7Lrs&list=PLgtmMKe4spCPsxyMpg-sxf3EcbsFYlzPK)

[Neural Networks and Deep Learning](https://github.com/mnielsen/neural-networks-and-deep-learning)
