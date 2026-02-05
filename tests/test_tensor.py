"""Tests for the Tensor implementation."""
import sys
sys.path.insert(0, '.')

from aloygrad.tensor import Tensor
from aloygrad.engine import Scalar


def test_tensor_creation():
    """Test creating tensors of different shapes."""
    # Scalar
    t = Tensor(5)
    assert t.shape == ()
    assert t.item() == 5
    
    # 1D
    t = Tensor([1, 2, 3])
    assert t.shape == (3,)
    
    # 2D
    t = Tensor([[1, 2], [3, 4]])
    assert t.shape == (2, 2)
    
    print("[PASS] Tensor creation tests passed")


def test_elementwise_ops():
    """Test element-wise operations."""
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    
    # Addition
    c = a + b
    assert c.shape == (3,)
    assert c[0].data == 5
    assert c[1].data == 7
    assert c[2].data == 9
    
    # Subtraction
    c = b - a
    assert c[0].data == 3
    assert c[1].data == 3
    assert c[2].data == 3
    
    # Multiplication
    c = a * b
    assert c[0].data == 4
    assert c[1].data == 10
    assert c[2].data == 18
    
    # Division
    c = b / a
    assert c[0].data == 4
    assert c[1].data == 2.5
    assert c[2].data == 2
    
    print("[PASS] Element-wise operation tests passed")


def test_unary_ops():
    """Test unary operations."""
    a = Tensor([1, -2, 3])
    
    # Negation
    b = -a
    assert b[0].data == -1
    assert b[1].data == 2
    assert b[2].data == -3
    
    # Power
    a = Tensor([2, 3, 4])
    b = a ** 2
    assert b[0].data == 4
    assert b[1].data == 9
    assert b[2].data == 16
    
    print("[PASS] Unary operation tests passed")


def test_activations():
    """Test activation functions."""
    a = Tensor([0, 1, -1])
    
    # ReLU
    b = a.relu()
    assert b[0].data == 0
    assert b[1].data == 1
    assert b[2].data == 0
    
    # Tanh
    b = a.tanh()
    assert abs(b[0].data - 0) < 1e-6
    assert abs(b[1].data - 0.7615941559557649) < 1e-6
    
    print("[PASS] Activation function tests passed")


def test_reductions():
    """Test reduction operations."""
    a = Tensor([[1, 2], [3, 4]])
    
    # Sum
    s = a.sum()
    assert isinstance(s, Scalar)
    assert s.data == 10
    
    # Mean
    m = a.mean()
    assert isinstance(m, Scalar)
    assert m.data == 2.5
    
    print("[PASS] Reduction operation tests passed")


def test_matmul():
    """Test matrix multiplication."""
    # 2D @ 2D
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    c = a @ b
    
    assert c.shape == (2, 2)
    assert c[0][0].data == 19  # 1*5 + 2*7
    assert c[0][1].data == 22  # 1*6 + 2*8
    assert c[1][0].data == 43  # 3*5 + 4*7
    assert c[1][1].data == 50  # 3*6 + 4*8
    
    # Vector @ Matrix
    v = Tensor([1, 2])
    m = Tensor([[1, 2, 3], [4, 5, 6]])
    result = v @ m
    assert result.shape == (1, 3)
    assert result[0][0].data == 9   # 1*1 + 2*4
    assert result[0][1].data == 12  # 1*2 + 2*5
    assert result[0][2].data == 15  # 1*3 + 2*6
    
    print("[PASS] Matrix multiplication tests passed")


def test_backward_simple():
    """Test backward pass on simple operations."""
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = a + b
    loss = c.sum()
    
    loss.backward()
    
    # Gradient should be 1 for all elements (sum derivative)
    assert a[0].grad == 1
    assert a[1].grad == 1
    assert a[2].grad == 1
    assert b[0].grad == 1
    assert b[1].grad == 1
    assert b[2].grad == 1
    
    print("[PASS] Simple backward pass tests passed")


def test_backward_complex():
    """Test backward pass on more complex computation."""
    # Create a simple computation: loss = (a * b).sum()
    a = Tensor([2, 3])
    b = Tensor([4, 5])
    c = a * b
    loss = c.sum()
    
    loss.backward()
    
    # d(loss)/da = b (since loss = sum(a*b))
    assert a[0].grad == 4
    assert a[1].grad == 5
    
    # d(loss)/db = a
    assert b[0].grad == 2
    assert b[1].grad == 3
    
    print("[PASS] Complex backward pass tests passed")


def test_matmul_backward():
    """Test backward pass through matrix multiplication."""
    # Simple linear transformation: y = W @ x
    W = Tensor([[1, 2], [3, 4]])
    x = Tensor([[5], [6]])
    y = W @ x
    
    # y = [[1*5 + 2*6], [3*5 + 4*6]] = [[17], [39]]
    assert y[0][0].data == 17
    assert y[1][0].data == 39
    
    loss = y.sum()
    loss.backward()
    
    # Verify gradients exist
    assert W[0][0].grad != 0
    assert x[0][0].grad != 0
    
    print("[PASS] Matmul backward pass tests passed")


def test_zero_grad():
    """Test gradient zeroing."""
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = a + b
    loss = c.sum()
    
    loss.backward()
    
    # Check gradients exist
    assert a[0].grad == 1
    
    # Zero gradients
    a.zero_grad()
    assert a[0].grad == 0
    assert a[1].grad == 0
    assert a[2].grad == 0
    
    print("[PASS] Zero grad tests passed")


def test_multilayer_backward():
    """Test backward through multiple operations."""
    # f(x) = ((x * 2) + 3) * 4
    x = Tensor([1, 2])
    y = x * 2        # [2, 4]
    z = y + Tensor([3, 3])  # [5, 7]
    w = z * 4        # [20, 28]
    loss = w.sum()   # 48
    
    assert loss.data == 48
    
    loss.backward()
    
    # df/dx = 4 * 2 = 8 for each element
    assert x[0].grad == 8
    assert x[1].grad == 8
    
    print("[PASS] Multi-layer backward pass tests passed")


def run_all_tests():
    """Run all tensor tests."""
    print("\n=== Running Tensor Tests ===\n")
    
    test_tensor_creation()
    test_elementwise_ops()
    test_unary_ops()
    test_activations()
    test_reductions()
    test_matmul()
    test_backward_simple()
    test_backward_complex()
    test_matmul_backward()
    test_zero_grad()
    test_multilayer_backward()
    
    print("\n=== All Tensor Tests Passed! ===\n")


if __name__ == "__main__":
    run_all_tests()
