# MLscratch

A machine learning library written in C from scratch — no PyTorch, no TensorFlow, no BLAS, no external math. Just `<math.h>`, raw memory, and the linear algebra needed to make a neural network learn to read handwritten digits.

I built this because I wanted to understand neural networks at the level where you can't hide behind an API. If you call `model.fit()`, you can pretend the gradients show up by magic. In C, you have to *be* the magic.

Inspired by and based on the tutorial by [Magicalbat](https://www.youtube.com/@Magicalbat): [Machine Learning in C](https://www.youtube.com/watch?v=hL_n_GljC0I).

---

## Background: How Neural Networks Actually Work

Before getting into the code, here's the full mental model you need. If you already know all this, skip to the next section.

### 1. The neuron

A neuron is the smallest unit of a neural network, and it is much simpler than the biology metaphor suggests. It takes some numbers in, multiplies each by a *weight*, adds them up, adds a *bias*, and passes the result through a nonlinear function.

$$y = f(w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b)$$

That's it. The weights $w_i$ control how much each input matters. The bias $b$ shifts the result up or down. The function $f$ is the *activation function* and exists for one reason: to break linearity (more on that in a moment).

If you stack a bunch of neurons that all see the same inputs, you get a **layer**. Mathematically, that whole layer becomes a single matrix multiplication:

$$\mathbf{z} = W \mathbf{x} + \mathbf{b}$$

where $W$ is the weight matrix (one row per neuron, one column per input), $\mathbf{x}$ is the input vector, and $\mathbf{b}$ is the bias vector. This is the single most important equation in deep learning, and you will see it implemented in this codebase as `mat_mul` followed by `mat_add`.

### 2. Why activation functions exist

If you stack two linear layers, $W_2(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$, you can multiply the matrices together and discover that the whole thing is just *one* linear function in disguise. A network of any depth would collapse into a single layer. Useless.

To actually compose layers into something more powerful, you need a nonlinearity between them. The two used in this project:

**ReLU** (Rectified Linear Unit) — used for hidden layers. It is comically simple:

$$\text{ReLU}(x) = \max(0, x)$$

Negative numbers become zero, positive numbers pass through. Despite looking like nothing, this works extremely well: it is fast, its gradient is trivial (1 if $x > 0$, else 0), and it doesn't saturate the way sigmoid/tanh do for large inputs.

**Softmax** — used for the output layer when you're doing classification. It takes a vector of arbitrary numbers and turns them into a probability distribution (positive numbers that sum to 1):

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

So if the network's last layer outputs `[2.0, 5.0, 1.0, ...]`, softmax converts that into something like `[0.04, 0.86, 0.02, ...]` — "I'm 86% sure it's class 1." This is exactly what you want for "is this digit a 0, 1, 2, ... or 9?"

### 3. The forward pass

Given an input — say, the 784 pixel values of a 28×28 MNIST image, flattened — the forward pass is just chaining the equations above:

$$\mathbf{a}_1 = \text{ReLU}(W_1 \mathbf{x} + \mathbf{b}_1)$$
$$\mathbf{a}_2 = \text{ReLU}(W_2 \mathbf{a}_1 + \mathbf{b}_2)$$
$$\mathbf{\hat{y}} = \text{softmax}(W_3 \mathbf{a}_2 + \mathbf{b}_3)$$

The output $\hat{\mathbf{y}}$ is a vector of 10 probabilities. The largest one is the network's guess.

### 4. Loss: how wrong are we?

To train, you need a number that says how badly the network screwed up on this example, so you can try to make it smaller. For classification with softmax outputs, the standard choice is **cross-entropy loss**:

$$L = -\sum_i y_i \log(\hat{y}_i)$$

where $\mathbf{y}$ is the one-hot true label (e.g. `[0,0,0,1,0,0,0,0,0,0]` for the digit 3) and $\hat{\mathbf{y}}$ is the network's predicted probabilities. Because $\mathbf{y}$ is one-hot, only one term in the sum survives — the one for the correct class — so the loss is simply $-\log(\hat{y}_\text{correct})$.

The intuition: if the network confidently says "3" and the answer is "3" (probability 0.99), the loss is tiny ($-\log 0.99 \approx 0.01$). If it confidently says "8" when the answer is "3" (probability 0.01 for the correct class), the loss is huge ($-\log 0.01 \approx 4.6$). Cross-entropy punishes confident wrong answers brutally, which is exactly what you want.

### 5. Learning = gradient descent

So we have a loss number. How do we make it smaller?

Every weight in the network is a knob. The loss is a function of all those knobs. If we knew which direction to turn each knob to lower the loss, we'd just do that. That direction is exactly what the **gradient** tells us: $\frac{\partial L}{\partial w}$ is "how much does the loss change if I nudge $w$ a tiny bit?"

The update rule, **gradient descent**, is:

$$w \leftarrow w - \eta \frac{\partial L}{\partial w}$$

where $\eta$ is the **learning rate**, a small number like 0.01. Subtract because we want to *decrease* the loss, and we move opposite to the gradient (the gradient points uphill; we want downhill).

Do this for every weight and every bias in the network, repeatedly, on lots of examples, and the network learns.

### 6. Backpropagation: where do gradients come from?

Computing $\frac{\partial L}{\partial w}$ for every single weight sounds horrifying — there are thousands of them, and the loss is a deeply nested function of all of them. The trick is **backpropagation**, which is just a clever application of the chain rule from calculus.

The chain rule says: if $L$ depends on $a$, and $a$ depends on $w$, then

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial w}$$

A neural network is a long chain of operations, so if you start at the loss and work *backwards*, multiplying local derivatives as you go, you can compute the gradient of the loss with respect to every weight in one sweep through the network. That's it. That's backprop.

In code, you build a graph of operations during the forward pass, then walk it in reverse, and at each node you know how to take the gradient flowing in from above and route it to the inputs of that node. The local rules:

- **Add** $c = a + b$: gradient flows unchanged to both inputs.
- **MatMul** $C = AB$: gradients are $\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} B^T$ and $\frac{\partial L}{\partial B} = A^T \frac{\partial L}{\partial C}$.
- **ReLU**: gradient passes through where the input was positive, zero otherwise.
- **Softmax + cross-entropy** (when paired): the combined gradient simplifies beautifully to $\hat{\mathbf{y}} - \mathbf{y}$ — predicted minus true. This is so clean it almost feels illegal.

### 7. Stochastic gradient descent and mini-batches

Computing the loss and gradient on the entire dataset (60,000 images) before updating weights once is slow and unnecessary. Instead we use **mini-batch SGD**: take a small random batch (e.g. 50 images), compute the average gradient on that batch, do one update, repeat. The randomness is actually helpful — it shakes the optimizer out of bad local minima.

One full pass through the training set is called an **epoch**. Train for several epochs, shuffle between them, and accuracy climbs.

That's the whole theory. Everything else in modern deep learning — convolutions, transformers, optimizers like Adam — is variations and improvements on these six ideas.

---

## How This Code Implements All of That

The codebase is small but does something architecturally interesting: it builds a tiny **autograd engine**. Instead of hardcoding the forward and backward passes for one specific network, it lets you describe a computation graph as data, and then it executes forward and backward passes generically. This is exactly how PyTorch and TensorFlow work conceptually, just radically simpler.

### The matrix

Everything is a `matrix` — a struct with `rows`, `cols`, and a `float*` to the data, stored row-major.

```c
typedef struct {
    uint32_t rows, cols;
    float *data;
} matrix;
```

On top of this struct sit all the primitive operations you'd expect: `mat_add`, `mat_sub`, `mat_mul`, `mat_scale`, `mat_relu`, `mat_softmax`, `mat_cross_entropy_loss`, plus the gradient-adding versions (`mat_softmax_add_grad`, `mat_relu_add_grad`, etc.). `mat_mul` supports transposing either operand on the fly, which is essential for backprop — the gradient through a matmul needs $A^T$ and $B^T$, and we don't want to materialize transposes.

Memory is managed with an **arena allocator** (`arena.c` / `arena.h`). You allocate a giant block once, push allocations into it linearly, and free everything at the end. There is no `malloc`/`free` ping-pong. This is fast, cache-friendly, and you can never leak.

### The autograd graph: `model_var`

The core idea: every quantity in the network — inputs, weights, biases, intermediate activations, the loss — is a `model_var`. Each `model_var` holds a value matrix, optionally a gradient matrix, an op code (what operation produced it), and pointers to its input `model_var`s.

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

When you write `mv_matmul(arena, model, W0, input, 0)`, you don't compute anything — you just create a new `model_var` whose op is `MV_OP_MATMUL` and whose inputs are `W0` and `input`. You're describing the graph. Building the network looks like writing math:

```c
model_var* z0_a = mv_matmul(arena, model, W0, input, 0);
model_var* z0_b = mv_add(arena, model, z0_a, b0, 0);
model_var* a0   = mv_relu(arena, model, z0_b, 0);
```

That's `a0 = ReLU(W0·x + b0)` — one layer of the network, expressed as graph nodes.

### Compile: turning the graph into a runnable program

Once you've built the graph, `model_compile` walks it from the output backwards and produces a flat **topological ordering** of nodes — a `model_program`. Topological order means: every node appears after the nodes it depends on. This means you can just iterate through the array left-to-right to do a forward pass, and right-to-left to do a backward pass. No recursion, no tree walking.

Two programs get compiled:
- `forward_prog`: produces the output (used for inference).
- `cost_prog`: produces the loss (used for training, since training needs gradients of the loss).

### Forward pass: `model_prog_compute`

Walk the program array forward. For each node, dispatch on its op and call the corresponding matrix function:

- `MV_OP_MATMUL` → `mat_mul`
- `MV_OP_ADD` → `mat_add`
- `MV_OP_RELU` → `mat_relu`
- `MV_OP_SOFTMAX` → `mat_softmax`
- `MV_OP_CROSS_ENTROPY` → `mat_cross_entropy_loss`

After the loop, every `model_var->val` is filled in, including the loss.

### Backward pass: `model_prog_compute_grads`

This is where backprop actually lives. The loss node's gradient with respect to itself is set to 1 (a vector of ones, in this case), and then we walk the program array in **reverse**. For each node, we use its op and the gradient already computed for it to push contributions back to its inputs' gradients:

```c
case MV_OP_MATMUL:
    if (a requires grad) mat_mul(a->grad, cur->grad, b->val, 0, 0, 1);  // dL/dA = dL/dC · B^T
    if (b requires grad) mat_mul(b->grad, a->val, cur->grad, 0, 1, 0);  // dL/dB = A^T · dL/dC

case MV_OP_ADD:
    // gradient passes through unchanged to both inputs
    mat_add(a->grad, a->grad, cur->grad);
    mat_add(b->grad, b->grad, cur->grad);

case MV_OP_RELU:
    mat_relu_add_grad(a->grad, a->val, cur->grad);  // zero out where input was negative

case MV_OP_SOFTMAX:
    mat_softmax_add_grad(a->grad, cur->val, cur->grad);  // multiply by Jacobian

case MV_OP_CROSS_ENTROPY:
    mat_cross_entropy_add_grad(p->grad, q->grad, p->val, q->val, cur->grad);
```

When the loop ends, every parameter (every weight and bias marked `MV_FLAG_PARAMETER | MV_FLAG_REQUIRES_GRAD`) has its gradient sitting in its `grad` matrix, ready to be subtracted from the value.

### The MNIST model

`create_mnist_model` builds the actual network. It's a 3-layer MLP with a small twist:

```
input        : 784 × 1     (flattened 28×28 image)
W0  (16×784) → b0  → ReLU  →  a0  (16 × 1)
W1  (16×16)  → b1  → ReLU  → +a0 →  a1  (16 × 1)   ← residual skip connection
W2  (10×16)  → b2  → softmax →  output (10 × 1)
cross_entropy(desired_output, output) → cost
```

Two things worth noting:

**Residual skip connection.** The line `model_var* a1 = mv_add(arena, model, a0, z1_c, 0);` adds the output of the first hidden layer's activations directly onto the second hidden layer's output. This is a miniature version of the skip connections from ResNet. It keeps gradients from vanishing in deeper networks — and even at depth 3, it helps.

**Xavier/Glorot weight initialization.** Weights are drawn uniformly from $[-\sqrt{6/(n_\text{in} + n_\text{out})}, +\sqrt{6/(n_\text{in} + n_\text{out})}]$:

```c
f32 bound0 = sqrtf(6.0f / (784 + 16));
mat_fill_rand(W0->val, -bound0, bound0);
```

This keeps the variance of activations roughly constant across layers at the start of training, which is the difference between a network that learns and a network whose values explode or collapse to zero on the first forward pass.

### The training loop: `model_train`

The training procedure follows the textbook recipe exactly:

1. **Shuffle** the training indices each epoch (a Fisher-Yates-ish shuffle using the project's PRNG).
2. For each **mini-batch** (size 50):
   - Zero out parameter gradients.
   - For each example in the batch: copy the image into `model->input`, copy the one-hot label into `model->desired_output`, run the forward+cost program, run the backward program. Gradients **accumulate** across the batch.
   - After the batch: scale gradients by `learning_rate / batch_size` (this both averages and applies the learning rate in one shot), then `param_value -= param_grad` for every parameter.
3. After each epoch, run the entire test set through the forward program and report accuracy and average loss.

Default hyperparameters: 3 epochs, batch size 50, learning rate 0.01.

### Data loading

MNIST is loaded from `.npy` files prepared by `scripts/nist.py` (which uses Keras to fetch MNIST and dumps it as numpy arrays of float32 in `[0, 1]`). The C loader skips the numpy header and reads the raw float bytes directly into matrices. Labels are converted from integers to one-hot vectors immediately on load.

There is also `draw_mnist_digit`, which prints a digit to the terminal using ANSI 256-color background escape codes. It is delightful and you should look at it.

---

## Results

After 3 epochs of training, the network typically reaches **~95% accuracy** on the MNIST test set. Not state-of-the-art — a 16-16-10 MLP doesn't have the capacity for that — but a clear demonstration that backprop, written by hand in C, learns.

The pre-training output is essentially uniform garbage (~0.1 for every class). The post-training output puts the bulk of its probability mass on the correct digit.

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

> Run from the `MLscratch` directory. Requires `git lfs pull` after cloning to download the MNIST data files.

## Dependencies

- C compiler (gcc / clang / MinGW)
- Git LFS (for data files)

## Resources Used

[3Blue1Brown Series](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=2)

[Adam Dhala](https://www.youtube.com/watch?v=Ixl3nykKG9M)

[Artem Kirsanov](https://www.youtube.com/watch?v=SmZmBKc7Lrs&list=PLgtmMKe4spCPsxyMpg-sxf3EcbsFYlzPK)

[Neural Networks and Deep Learning](https://github.com/mnielsen/neural-networks-and-deep-learning)
