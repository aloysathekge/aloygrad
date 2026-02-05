from aloygrad.engine import Scalar


def _to_scalar(x):
    """Convert a scalar to Scalar if needed."""
    if isinstance(x, Scalar):
        return x
    if isinstance(x, (int, float)):
        return Scalar(x)
    raise TypeError(f"Unsupported scalar type: {type(x)}")


def _to_scalars_tree(x):
    """Recursively convert nested lists/tuples to nested Scalars."""
    if isinstance(x, Tensor):
        return x.data
    if isinstance(x, (list, tuple)):
        return [_to_scalars_tree(v) for v in x]
    return _to_scalar(x)


def _map_unary(x, fn):
    """Apply a unary function element-wise to a nested structure."""
    if isinstance(x, Scalar):
        return fn(x)
    return [_map_unary(v, fn) for v in x]


def _map_binary(a, b, fn):
    """Apply a binary function element-wise to two nested structures."""
    if isinstance(a, Scalar) and isinstance(b, Scalar):
        return fn(a, b)
    # Broadcast scalar to match list structure
    if isinstance(a, Scalar) and isinstance(b, list):
        return [_map_binary(a, item, fn) for item in b]
    if isinstance(a, list) and isinstance(b, Scalar):
        return [_map_binary(item, b, fn) for item in a]
    # Both are lists
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise ValueError("Shape mismatch for elementwise op")
        return [_map_binary(x, y, fn) for x, y in zip(a, b)]
    raise TypeError("Shape mismatch for elementwise op")


def _is_2d(x):
    """Check if x represents a 2D structure."""
    return isinstance(x, list) and (len(x) == 0 or isinstance(x[0], list))


def _as_2d(x):
    """Ensure x is 2D - if it's 1D, treat as a single row."""
    if isinstance(x, list) and (len(x) == 0 or isinstance(x[0], Scalar)):
        return [x]
    return x


def _shape_2d(x):
    """Get the shape of a 2D nested list structure."""
    if not _is_2d(x):
        raise ValueError("Expected 2D data for matmul")
    rows = len(x)
    cols = 0 if rows == 0 else len(x[0])
    for r in x:
        if len(r) != cols:
            raise ValueError("Ragged 2D data")
    return rows, cols


def _flatten(x):
    """Flatten a nested structure to a list of Scalars."""
    if isinstance(x, Scalar):
        return [x]
    out = []
    for v in x:
        out.extend(_flatten(v))
    return out


def _infer_shape(x):
    """Infer the shape of a nested structure."""
    if isinstance(x, Scalar):
        return ()  # scalar
    if not x:
        return (0,)
    if isinstance(x[0], Scalar):
        return (len(x),)  # 1D
    # 2D case
    rows = len(x)
    cols = len(x[0]) if x else 0
    # Verify all rows same length
    for r in x:
        if len(r) != cols:
            raise ValueError("Ragged array - all rows must have same length")
    return (rows, cols)


class Tensor:
    """
    A tensor built from nested lists of scalar Scalar objects.
    Supports basic operations with automatic differentiation.
    """
    
    def __init__(self, data):
        """
        Create a tensor from nested lists, scalars, or another Tensor.
        
        Examples:
            Tensor(5)           # scalar
            Tensor([1, 2, 3])   # 1D tensor
            Tensor([[1, 2], [3, 4]])  # 2D tensor
        """
        self.data = _to_scalars_tree(data)

    @property
    def shape(self):
        """Return the shape of this tensor as a tuple."""
        return _infer_shape(self.data)

    def __repr__(self):
        return f"Tensor(shape={self.shape})"

    def __iter__(self):
        """Allow iteration over the first dimension."""
        return iter(self.data)

    def __getitem__(self, idx):
        """Access elements by index."""
        return self.data[idx]

    def item(self):
        """Extract the scalar value (only for scalar tensors)."""
        if isinstance(self.data, Scalar):
            return self.data.data
        raise ValueError("item() only works on scalar tensors")

    def __add__(self, other):
        """Element-wise addition."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(_map_binary(self.data, other.data, lambda a, b: a + b))

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        """Element-wise subtraction."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(_map_binary(self.data, other.data, lambda a, b: a - b))

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(_map_binary(other.data, self.data, lambda a, b: a - b))

    def __mul__(self, other):
        """Element-wise multiplication."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(_map_binary(self.data, other.data, lambda a, b: a * b))

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        """Element-wise division."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(_map_binary(self.data, other.data, lambda a, b: a / b))

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(_map_binary(other.data, self.data, lambda a, b: a / b))

    def __neg__(self):
        """Element-wise negation."""
        return Tensor(_map_unary(self.data, lambda a: -a))

    def __pow__(self, p):
        """Element-wise power."""
        return Tensor(_map_unary(self.data, lambda a: a ** p))

    def pow(self, p):
        """Element-wise power (method form)."""
        return self ** p

    def tanh(self):
        """Element-wise tanh activation."""
        return Tensor(_map_unary(self.data, lambda a: a.tanh()))

    def relu(self):
        """Element-wise ReLU activation."""
        return Tensor(_map_unary(self.data, lambda a: a.relu()))

    def sigmoid(self):
        """Element-wise sigmoid activation."""
        return Tensor(_map_unary(self.data, lambda a: a.sigmoid()))

    def sum(self):
        """Sum all elements, returning a scalar Scalar."""
        vals = _flatten(self.data)
        if not vals:
            return Scalar(0.0)
        out = vals[0]
        for v in vals[1:]:
            out = out + v
        return out

    def mean(self):
        """Compute the mean of all elements, returning a scalar Scalar."""
        vals = _flatten(self.data)
        if not vals:
            return Scalar(0.0)
        return self.sum() * (1.0 / len(vals))

    def matmul(self, other):
        """
        Matrix multiplication.
        
        Supports:
        - 2D @ 2D -> 2D
        - 1D @ 2D -> 1D (treated as row vector)
        - 2D @ 1D -> 1D (treated as column vector)
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        a = _as_2d(self.data)
        b = _as_2d(other.data)
        a_rows, a_cols = _shape_2d(a)
        b_rows, b_cols = _shape_2d(b)
        
        if a_cols != b_rows:
            raise ValueError(f"matmul shape mismatch: ({a_rows}, {a_cols}) @ ({b_rows}, {b_cols})")

        out = []
        for i in range(a_rows):
            row = []
            for j in range(b_cols):
                acc = None
                for k in range(a_cols):
                    prod = a[i][k] * b[k][j]
                    acc = prod if acc is None else acc + prod
                row.append(acc if acc is not None else Scalar(0.0))
            out.append(row)
        return Tensor(out)

    def __matmul__(self, other):
        """Matrix multiplication operator (@)."""
        return self.matmul(other)

    def backward(self):
        """
        Perform backpropagation through the computation graph.
        For non-scalar tensors, sums all elements before backprop.
        """
        if isinstance(self.data, Scalar):
            self.data.backward()
        else:
            self.sum().backward()

    def zero_grad(self):
        """Zero out all gradients in this tensor."""
        vals = _flatten(self.data)
        for v in vals:
            v.grad = 0.0
