# AloyGrad

A deep learning framework built from scratch for educational purposes. AloyGrad implements automatic differentiation, tensor operations, and neural network building blocks—all without relying on PyTorch or TensorFlow.

## Features

- **Scalar Autograd Engine** (`Scalar`): micrograd-style automatic differentiation for scalars
- **Tensor System** (`Tensor`): Multi-dimensional arrays built on top of scalar autograd
- **Neural Network Modules**: Basic building blocks for creating neural networks
- **Loss Functions**: MSE loss for regression tasks
- **Optimizers**: SGD optimizer with gradient descent

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd aloygrad
```

No external dependencies required for core functionality!

## Quick Start

### Scalar Autograd

```python
from aloygrad import Value

# Create values
a = Value(2.0)
b = Value(3.0)

# Build computation graph
c = a * b + b ** 2
print(f"c = {c.data}")  # 15.0

# Compute gradients
c.backward()
print(f"dc/da = {a.grad}")  # 3.0
print(f"dc/db = {b.grad}")  # 8.0
```

### Tensor Operations

```python
from aloygrad import Tensor

# Create tensors
a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])

# Element-wise operations
c = a + b  # [5, 7, 9]
d = a * b  # [4, 10, 18]

# Reductions
total = c.sum()  # 21
avg = c.mean()    # 7

# Matrix multiplication
A = Tensor([[1, 2], [3, 4]])
B = Tensor([[5, 6], [7, 8]])
C = A @ B  # Matrix product

# Automatic differentiation
loss = c.sum()
loss.backward()
print([x.grad for x in a.data])  # [1.0, 1.0, 1.0]
```

### Neural Networks

```python
from aloygrad.nn import MLP

# Create a multi-layer perceptron
model = MLP(nin=3, nouts=[4, 4, 1])

# Forward pass
x = [1.0, 2.0, 3.0]
y = model(x)

# Get parameters
params = model.parameters()
print(f"Total parameters: {len(params)}")
```

## Project Structure

```
aloygrad/
├── engine.py      # Scalar with autograd
├── tensor.py      # Tensor operations built on Scalar
├── nn.py          # Neural network modules
├── loss.py        # Loss functions
└── optimizer.py   # Optimization algorithms

examples/
└── tensor_demo.py # Comprehensive tensor demo

tests/
├── test_engine.py # Tests for scalar autograd
└── test_tensor.py # Tests for tensor operations
```

## Running Tests

```bash
# Test scalar autograd
python tests/test_engine.py

# Test tensor operations
python tests/test_tensor.py
```

## Running Examples

```bash
# Comprehensive tensor demonstration
python examples/tensor_demo.py
```

## Architecture

AloyGrad is structured in layers:

1. **Scalar Autograd (`Scalar`)**: Core automatic differentiation for individual scalars
   - Supports basic operations: `+`, `-`, `*`, `/`, `**`
   - Activation functions: `relu`, `tanh`, `sigmoid`
   - Backward pass via topological sort

2. **Tensor System (`Tensor`)**: Multi-dimensional arrays built from nested `Scalar` objects
   - Element-wise operations with scalar broadcasting
   - Matrix multiplication (`matmul`/`@`)
   - Reductions: `sum`, `mean`
   - Automatic differentiation through tensor operations

3. **Neural Network Modules (`nn`)**: High-level building blocks
   - `Module`: Base class for all neural network components
   - `Neuron`: Single neuron with weights and bias
   - `Layer`: Collection of neurons
   - `MLP`: Multi-layer perceptron

4. **Training Utilities**: Loss functions and optimizers
   - `MSELoss`: Mean squared error for regression
   - `SGD`: Stochastic gradient descent

## Educational Goals

This framework is designed to help understand:

- How automatic differentiation works (reverse-mode autodiff)
- How tensors can be built from scalar operations
- How neural networks are structured and trained
- The relationship between forward and backward passes
- Why frameworks like PyTorch are designed the way they are

## Design Decisions

### Why Nested Lists Instead of NumPy?

AloyGrad uses nested Python lists of `Scalar` objects rather than NumPy arrays. This approach:

- **Maintains transparency**: You can inspect the entire computation graph
- **Simplifies learning**: No hidden NumPy magic obscuring the autograd
- **Builds on scalars**: Everything reduces to scalar operations you already understand

The tradeoff is performance—but that's intentional. This is a learning framework, not a production system.

### Why No Broadcasting (Initially)?

Broadcasting adds complexity that can obscure core concepts. AloyGrad includes scalar-to-tensor broadcasting but requires explicit shape matching for most operations. This forces you to think about tensor shapes explicitly.

## Limitations

AloyGrad is for **education**, not production:

- No GPU acceleration
- Limited to small models and datasets
- Performance is not optimized
- Limited operation support compared to PyTorch/TensorFlow

## Future Enhancements

Potential additions to explore:

- [ ] More activation functions (Swish, GELU, etc.)
- [ ] Batch normalization and layer normalization
- [ ] Convolutional layers
- [ ] Recurrent layers (RNN, LSTM, GRU)
- [ ] More optimizers (Adam, RMSprop, AdaGrad)
- [ ] Cross-entropy loss
- [ ] Data loading utilities
- [ ] Model serialization (save/load)
- [ ] Full broadcasting support
- [ ] Gradient clipping
- [ ] Learning rate schedules

## Inspiration

This project is inspired by:

- [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy
- [PyTorch](https://pytorch.org/) design philosophy
- The desire to deeply understand how ML frameworks work

## License

MIT License - feel free to use for learning!

## Contributing

This is an educational project. Feel free to fork and experiment! If you find bugs or have suggestions for making it more educational, please open an issue.
