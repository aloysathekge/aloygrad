"""Tests for the scalar Scalar autograd engine."""
import sys
sys.path.insert(0, '.')

from aloygrad.engine import Scalar
import math


def test_value_creation():
    """Test creating Scalar objects."""
    a = Scalar(5)
    assert a.data == 5
    assert a.grad == 0
    print("[PASS] Scalar creation test passed")


def test_addition():
    """Test addition operation."""
    a = Scalar(2)
    b = Scalar(3)
    c = a + b
    assert c.data == 5
    
    c.backward()
    assert a.grad == 1
    assert b.grad == 1
    print("[PASS] Addition test passed")


def test_multiplication():
    """Test multiplication operation."""
    a = Scalar(2)
    b = Scalar(3)
    c = a * b
    assert c.data == 6
    
    c.backward()
    assert a.grad == 3  # dc/da = b
    assert b.grad == 2  # dc/db = a
    print("[PASS] Multiplication test passed")


def test_power():
    """Test power operation."""
    a = Scalar(2)
    b = a ** 3
    assert b.data == 8
    
    b.backward()
    assert a.grad == 12  # d(x^3)/dx = 3*x^2 = 3*4 = 12
    print("[PASS] Power test passed")


def test_division():
    """Test division operation."""
    a = Scalar(6)
    b = Scalar(2)
    c = a / b
    assert c.data == 3
    
    c.backward()
    assert a.grad == 0.5  # dc/da = 1/b
    assert b.grad == -1.5  # dc/db = -a/b^2 = -6/4
    print("[PASS] Division test passed")


def test_subtraction():
    """Test subtraction operation."""
    a = Scalar(5)
    b = Scalar(3)
    c = a - b
    assert c.data == 2
    
    c.backward()
    assert a.grad == 1
    assert b.grad == -1
    print("[PASS] Subtraction test passed")


def test_negation():
    """Test negation operation."""
    a = Scalar(5)
    b = -a
    assert b.data == -5
    
    b.backward()
    assert a.grad == -1
    print("[PASS] Negation test passed")


def test_relu():
    """Test ReLU activation."""
    # Positive input
    a = Scalar(2)
    b = a.relu()
    assert b.data == 2
    b.backward()
    assert a.grad == 1
    
    # Negative input
    c = Scalar(-2)
    d = c.relu()
    assert d.data == 0
    d.backward()
    assert c.grad == 0
    
    print("[PASS] ReLU test passed")


def test_tanh():
    """Test tanh activation."""
    a = Scalar(0)
    b = a.tanh()
    assert abs(b.data - 0) < 1e-6
    
    a = Scalar(1)
    b = a.tanh()
    expected = (math.exp(2) - 1) / (math.exp(2) + 1)
    assert abs(b.data - expected) < 1e-6
    
    b.backward()
    # d(tanh(x))/dx = 1 - tanh^2(x)
    expected_grad = 1 - expected ** 2
    assert abs(a.grad - expected_grad) < 1e-6
    
    print("[PASS] Tanh test passed")


def test_sigmoid():
    """Test sigmoid activation."""
    a = Scalar(0)
    b = a.sigmoid()
    assert abs(b.data - 0.5) < 1e-6
    
    a = Scalar(2)
    b = a.sigmoid()
    expected = 1 / (1 + math.exp(-2))
    assert abs(b.data - expected) < 1e-6
    
    print("[PASS] Sigmoid test passed")


def test_complex_expression():
    """Test a complex computational graph."""
    # f(x, y) = (x * y) + (x ** 2)
    x = Scalar(3)
    y = Scalar(4)
    
    z1 = x * y      # 12
    z2 = x ** 2     # 9
    f = z1 + z2     # 21
    
    assert f.data == 21
    
    f.backward()
    
    # df/dx = y + 2*x = 4 + 6 = 10
    assert x.grad == 10
    # df/dy = x = 3
    assert y.grad == 3
    
    print("[PASS] Complex expression test passed")


def test_chain_rule():
    """Test chain rule through multiple operations."""
    # f(x) = ((x * 2) + 1) ** 2
    x = Scalar(3)
    y = x * 2       # 6
    z = y + 1       # 7
    w = z ** 2      # 49
    
    assert w.data == 49
    
    w.backward()
    
    # df/dx = 2 * (x*2 + 1) * 2 = 4 * (6 + 1) = 28
    assert x.grad == 28
    
    print("[PASS] Chain rule test passed")


def test_multiple_uses():
    """Test when a value is used multiple times."""
    # f(x) = x + x + x = 3x
    x = Scalar(2)
    y = x + x + x
    
    assert y.data == 6
    
    y.backward()
    assert x.grad == 3  # df/dx = 3
    
    print("[PASS] Multiple uses test passed")


def test_reverse_ops():
    """Test reverse operations (radd, rmul, etc)."""
    a = Scalar(3)
    
    # Test radd
    b = 5 + a
    assert b.data == 8
    
    # Test rmul
    c = 2 * a
    assert c.data == 6
    
    # Test rsub
    d = 10 - a
    assert d.data == 7
    
    # Test rtruediv
    e = 12 / a
    assert e.data == 4
    
    print("[PASS] Reverse operations test passed")


def run_all_tests():
    """Run all engine tests."""
    print("\n=== Running Engine Tests ===\n")
    
    test_value_creation()
    test_addition()
    test_multiplication()
    test_power()
    test_division()
    test_subtraction()
    test_negation()
    test_relu()
    test_tanh()
    test_sigmoid()
    test_complex_expression()
    test_chain_rule()
    test_multiple_uses()
    test_reverse_ops()
    
    print("\n=== All Engine Tests Passed! ===\n")


if __name__ == "__main__":
    run_all_tests()
