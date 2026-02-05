import math
class Scalar:
    def __init__(self,data,_children=(),_op=''):
        self.data=data
        self.grad=0.0
        self._prev=set(_children)
        self._op=_op
        self._backward=lambda: None


    def __add__(self,other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out=Scalar(self.data +other.data,(self,other) , '+')
        
        def _backward():
            # Gradient flows equally to both inputs
            self.grad+=1.0 *out.grad
            other.grad +=1.0 * out.grad
        out._backward=_backward
        return out
    def __sub__(self, other):
        return self +(-other)
    
    def __rsub__(self,other):
         other = other if isinstance(other, Scalar) else Scalar(other)
         return other +(-self)

    def __radd__(self,other):

        other = other if isinstance(other, Scalar) else Scalar(other)
        return self +other
    
    def __neg__(self):
        return self*-1
    
   

    
    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __mul__(self,other):
        other= other if isinstance(other, Scalar) else Scalar(other)
        out=Scalar(self.data * other.data, (self,other), '*')

        def _backward():
            # Chain rule d(a*b)/da=b,d(a*b)/db=a
            self.grad +=other.data *out.grad
            other.grad +=self.data * out.grad
        out._backward=_backward
        return out

    def __pow__(self,other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Scalar(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out



    def __rmul__(self,other):
        return self * other

    def tanh(self):
        x=self.data
        t=(math.exp(2*x)-1)/(math.exp(2*x) +1)
        out= Scalar(t,(self,),'tanh')

        def _backward():
            #d(tanh(x))/dx=1-tanh**2(x)
            self.grad+=(1-t**2)*out.grad
        out._backward=_backward

        return out
    
    def relu(self):
        #Relu: max(0,x)

        out = Scalar(max(0,self.data),(self,),"Relu")

        def _backward():
            self.grad +=(1.0 if self.data>0 else 0.0)*out.grad
        out._backward=_backward
        return out


    
    def sigmoid(self):
        x=self.data
        s= 1/(1+math.exp(-x))
        out=Scalar(s,(self,),'sigmoid')

        def _backward():
            self.grad+=(s*(1+s))*out.grad
        out._backward=_backward
        return out

    

    def backward(self):
            #Build topological order of all nodes
            topo=[]
            visited =set()

            def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._prev:
                        build_topo(child)
                    topo.append(v)

            build_topo(self)

            # Backpropagation gradeints in reverse order

            self.grad=1.0

            for node in reversed(topo):
             node._backward()

