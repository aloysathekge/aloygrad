"""
Demonstration of the Tensor API in AloyGrad.
Shows basic operations, activations, and automatic differentiation.
"""
import sys
sys.path.insert(0, '.')

from aloygrad import Tensor, Scalar


def demo_basic_operations():
    """Demonstrate basic tensor operations."""
    print("\n=== Basic Tensor Operations ===\n")
    
    # Create tensors
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    
    print(f"a = {a} -> shape {a.shape}")
    print(f"b = {b} -> shape {b.shape}")
    
    # Element-wise operations
    c = a + b
    print(f"\na + b = Tensor with shape {c.shape}")
    print(f"Values: {[x.data for x in c.data]}")
    
    d = a * b
    print(f"\na * b = Tensor with shape {d.shape}")
    print(f"Values: {[x.data for x in d.data]}")
    
    # Scalar operations (broadcasting)
    e = a * 2
    print(f"\na * 2 = Tensor with shape {e.shape}")
    print(f"Values: {[x.data for x in e.data]}")


def demo_matrix_operations():
    """Demonstrate matrix operations."""
    print("\n=== Matrix Operations ===\n")
    
    # Matrix multiplication
    A = Tensor([[1, 2], [3, 4]])
    B = Tensor([[5, 6], [7, 8]])
    
    print(f"A = Tensor with shape {A.shape}")
    print(f"B = Tensor with shape {B.shape}")
    
    C = A @ B
    print(f"\nA @ B = Tensor with shape {C.shape}")
    print("Values:")
    for i in range(2):
        print(f"  [{C[i][0].data}, {C[i][1].data}]")


def demo_activations():
    """Demonstrate activation functions."""
    print("\n=== Activation Functions ===\n")
    
    x = Tensor([-2, -1, 0, 1, 2])
    print(f"x = Tensor with shape {x.shape}")
    print(f"Values: {[v.data for v in x.data]}")
    
    # ReLU
    relu_out = x.relu()
    print(f"\nReLU(x):")
    print(f"Values: {[v.data for v in relu_out.data]}")
    
    # Tanh
    tanh_out = x.tanh()
    print(f"\nTanh(x):")
    print(f"Values: {[round(v.data, 4) for v in tanh_out.data]}")
    
    # Sigmoid
    sigmoid_out = x.sigmoid()
    print(f"\nSigmoid(x):")
    print(f"Values: {[round(v.data, 4) for v in sigmoid_out.data]}")


def demo_reductions():
    """Demonstrate reduction operations."""
    print("\n=== Reduction Operations ===\n")
    
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    print(f"x = Tensor with shape {x.shape}")
    
    # Sum
    total = x.sum()
    print(f"\nsum(x) = {total.data}")
    
    # Mean
    avg = x.mean()
    print(f"mean(x) = {avg.data}")


def demo_autograd():
    """Demonstrate automatic differentiation."""
    print("\n=== Automatic Differentiation ===\n")
    
    # Simple example: loss = (a + b).sum()
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    
    print("Forward pass:")
    c = a + b
    loss = c.sum()
    print(f"a = {[x.data for x in a.data]}")
    print(f"b = {[x.data for x in b.data]}")
    print(f"c = a + b = {[x.data for x in c.data]}")
    print(f"loss = sum(c) = {loss.data}")
    
    print("\nBackward pass:")
    loss.backward()
    print(f"d(loss)/d(a) = {[x.grad for x in a.data]}")
    print(f"d(loss)/d(b) = {[x.grad for x in b.data]}")


def demo_complex_computation():
    """Demonstrate a more complex computation graph."""
    print("\n=== Complex Computation Graph ===\n")
    
    # Simulate a simple neural network computation
    # output = relu(W @ x + b)
    W = Tensor([[0.5, -0.3], [0.2, 0.8]])
    x = Tensor([[1], [2]])
    b = Tensor([[0.1], [-0.2]])
    
    print("Simulating: output = ReLU(W @ x + b)")
    print(f"W shape: {W.shape}")
    print(f"x shape: {x.shape}")
    print(f"b shape: {b.shape}")
    
    # Forward pass
    z = W @ x
    print(f"\nz = W @ x, shape: {z.shape}")
    
    z_plus_b = z + b
    print(f"z + b, shape: {z_plus_b.shape}")
    
    output = z_plus_b.relu()
    print(f"output = ReLU(z + b), shape: {output.shape}")
    print(f"Output values: {[output[i][0].data for i in range(2)]}")
    
    # Compute loss and backprop
    loss = output.sum()
    print(f"\nloss = sum(output) = {loss.data}")
    
    print("\nComputing gradients...")
    loss.backward()
    
    print(f"d(loss)/d(W[0][0]) = {W[0][0].grad}")
    print(f"d(loss)/d(x[0][0]) = {x[0][0].grad}")
    print(f"d(loss)/d(b[0][0]) = {b[0][0].grad}")


def demo_training_iteration():
    """Demonstrate a simple training iteration."""
    print("\n=== Simple Training Iteration ===\n")
    
    # Simple linear model: y = W @ x
    W = Tensor([[2.0, 3.0]])
    x = Tensor([[1.0], [1.0]])
    target = Tensor([[10.0]])
    
    learning_rate = 0.01
    
    print("Goal: Learn W such that W @ x ~= target")
    print(f"Initial W: {[W[0][i].data for i in range(2)]}")
    print(f"x: {[x[i][0].data for i in range(2)]}")
    print(f"target: {target[0][0].data}")
    
    for epoch in range(5):
        # Forward pass
        pred = W @ x
        loss = ((pred - target) ** 2).sum()
        
        # Backward pass
        loss.backward()
        
        # Update weights (simple SGD)
        for i in range(2):
            W[0][i].data -= learning_rate * W[0][i].grad
            W[0][i].grad = 0.0  # Zero gradients
        
        if epoch == 0 or (epoch + 1) % 1 == 0:
            print(f"\nEpoch {epoch + 1}: loss = {loss.data:.6f}")
            print(f"  W = {[round(W[0][i].data, 4) for i in range(2)]}")
            print(f"  pred = {pred[0][0].data:.4f}")


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("AloyGrad Tensor System Demo")
    print("="*60)
    
    demo_basic_operations()
    demo_matrix_operations()
    demo_activations()
    demo_reductions()
    demo_autograd()
    demo_complex_computation()
    demo_training_iteration()
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
