# AloyGrad Quick Start Guide

## Installation

No installation needed! Just clone and use:

```bash
cd aloygrad
python  # Start Python in the project directory
```

## 5-Minute Tutorial

### 1. Scalar Autograd

```python
from aloygrad import Scalar

# Simple computation
a = Scalar(2.0)
b = Scalar(3.0)
c = a * b + b ** 2  # 2*3 + 3^2 = 15

# Compute gradients
c.backward()
print(f"dc/da = {a.grad}")  # 3.0
print(f"dc/db = {b.grad}")  # 8.0
```

### 2. Tensor Operations

```python
from aloygrad import Tensor

# Create and operate on tensors
x = Tensor([1, 2, 3])
y = Tensor([4, 5, 6])
z = x + y  # Element-wise addition

# Automatic differentiation
loss = z.sum()
loss.backward()
print([v.grad for v in x.data])  # [1.0, 1.0, 1.0]
```

### 3. Matrix Multiplication

```python
# Matrix operations
A = Tensor([[1, 2], [3, 4]])
B = Tensor([[5, 6], [7, 8]])
C = A @ B  # or A.matmul(B)

print(C.shape)  # (2, 2)
```

### 4. Neural Network

```python
from aloygrad.nn import MLP

# Create a 3-layer network
model = MLP(nin=3, nouts=[4, 4, 1])

# Forward pass
x = [1.0, 2.0, 3.0]
output = model(x)

# Get all parameters
params = model.parameters()
```

### 5. Training Loop

```python
from aloygrad import Tensor
from aloygrad.loss import MSELoss
from aloygrad.optimizer import SGD

# Setup
model = MLP(nin=2, nouts=[4, 1])
loss_fn = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01)

# Training iteration
for epoch in range(100):
    # Forward
    pred = model([1.0, 2.0])
    loss = loss_fn([pred], [3.0])
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss = {loss.data}")
```

## Running Examples

```bash
# Comprehensive tensor demo
python examples/tensor_demo.py

# Run tests
python tests/test_engine.py
python tests/test_tensor.py
```

## Common Patterns

### Creating Tensors

```python
# Scalar
t = Tensor(5)

# 1D (vector)
t = Tensor([1, 2, 3])

# 2D (matrix)
t = Tensor([[1, 2], [3, 4]])

# From existing tensor
t2 = Tensor(t)
```

### Accessing Data

```python
t = Tensor([1, 2, 3])

# Shape
print(t.shape)  # (3,)

# Individual elements
print(t[0].data)  # 1

# Scalar value
scalar = Tensor(5)
print(scalar.item())  # 5
```

### Operations

```python
a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])

# Element-wise
c = a + b    # Addition
c = a - b    # Subtraction
c = a * b    # Multiplication
c = a / b    # Division
c = a ** 2   # Power
c = -a       # Negation

# Matrix
A = Tensor([[1, 2], [3, 4]])
B = Tensor([[5, 6], [7, 8]])
C = A @ B    # Matrix multiplication

# Activations
c = a.relu()
c = a.tanh()
c = a.sigmoid()

# Reductions
s = a.sum()   # Returns Scalar, not Tensor
m = a.mean()  # Returns Scalar, not Tensor
```

### Gradients

```python
# Compute gradients
loss.backward()

# Access gradients (for Scalar)
print(scalar.grad)

# Access gradients (for Tensor)
for v in tensor.data:
    print(v.grad)  # if 1D
# or
for row in tensor.data:
    for v in row:
        print(v.grad)  # if 2D

# Zero gradients
tensor.zero_grad()
# or
optimizer.zero_grad()
```

### Building Models

```python
from aloygrad.nn import Module, Layer, Neuron

# Custom module
class MyModel(Module):
    def __init__(self):
        self.layer1 = Layer(3, 4)
        self.layer2 = Layer(4, 1, nonlin=False)
    
    def __call__(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    
    def parameters(self):
        return (self.layer1.parameters() + 
                self.layer2.parameters())

# Use it
model = MyModel()
output = model([1.0, 2.0, 3.0])
```

## Tips & Tricks

### 1. Debugging Gradients

```python
# Check if gradients are flowing
loss.backward()
for i, p in enumerate(model.parameters()):
    print(f"Param {i}: grad = {p.grad}")
```

### 2. Understanding Shapes

```python
# Always check shapes when debugging
print(f"Input: {x.shape}")
print(f"Weight: {W.shape}")
print(f"Output: {y.shape}")
```

### 3. Scalar vs Tensor

```python
# Reductions return Scalar, not Tensor
loss = tensor.sum()  # Returns Scalar
print(type(loss))    # <class 'Scalar'>

# To get python number
print(loss.data)     # float

# For scalar tensor
scalar_tensor = Tensor(5)
print(scalar_tensor.item())  # 5
```

### 4. Manual Training Loop

```python
# Full control
for epoch in range(epochs):
    # Zero gradients
    for p in model.parameters():
        p.grad = 0.0
    
    # Forward
    output = model(input)
    loss = compute_loss(output, target)
    
    # Backward
    loss.backward()
    
    # Update
    for p in model.parameters():
        p.data -= learning_rate * p.grad
```

## Troubleshooting

### Shape Mismatch Error

```python
# Error: "Shape mismatch for elementwise op"
# Solution: Ensure tensors have same shape
a = Tensor([1, 2, 3])
b = Tensor([4, 5])  # Different length!
c = a + b  # ERROR

# Fix: Match shapes
b = Tensor([4, 5, 6])
c = a + b  # OK
```

### No Gradients

```python
# Problem: Gradients are zero
# Solutions:
# 1. Did you call backward()?
loss.backward()

# 2. Did you zero gradients too early?
# (Zero BEFORE forward/backward, not after)

# 3. Did you use operations that break the graph?
# (Use Scalar/Tensor operations, not raw Python)
```

### Matrix Multiplication Error

```python
# Error: "matmul shape mismatch"
# Solution: Check inner dimensions match
A = Tensor([[1, 2, 3]])    # (1, 3)
B = Tensor([[4, 5]])        # (1, 2) - wrong!
C = A @ B  # ERROR: 3 != 1

# Fix:
B = Tensor([[4], [5], [6]])  # (3, 1)
C = A @ B  # OK: (1, 3) @ (3, 1) = (1, 1)
```

## Next Steps

1. **Read the README** for architecture overview
2. **Run examples/tensor_demo.py** to see all features
3. **Read tests/** to see how everything works
4. **Build your own model** - start simple!
5. **Experiment** - break things and learn!

## Resources

- `README.md` - Full documentation
- `IMPLEMENTATION_SUMMARY.md` - Technical details
- `examples/tensor_demo.py` - Comprehensive examples
- `tests/` - Usage examples in tests
