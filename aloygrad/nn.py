import random
from value import Value


class Module:
    def parameters(self):
        
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad=0

class Neuron(Module):
    def __init__(self,nin,nonlin=True):
        self.w=[Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b=Value(0)

        self.nonlin=nonlin
    
    def __call__(self,x) :
        act=sum((wi*xi for wi,xi in zip(self.w,x)),self.b)

        return act.relu() if self.nonlin else act
    def parameters(self):
        return self.w+[self.b]
    def __repr__(self):
         return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
    def __init__(self,nin,nout,**kwargs):
        # nin number of inputs 
        # nout = number of neurons in this layer
        # **kwargs = extra args like nonlin being True/False

        self.neurons=[Neuron(nin,**kwargs) for _ in range(nout)]
    
    def __call__(self,x):
        # feed x to each neuron

        out=[n(x) for n in self.neurons]
        # if only 1 neuron , return the value (not a list)
        return out[0] if len(out)==1 else out

    def parameters(self):
        # collect parameters from ALLneurons
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self) -> str:
        return f"Layer of [{','.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    def __init__(self,nin,nouts):

        sz=[nin]+nouts
        self.layers=[Layer(sz[i],sz[i+1],nonlin=(i !=len(nouts)-1)) for i in range(len(nouts))]

    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for  p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

class MSELoss:
    """Mean Squared Error for regression"""
    def __call__(self, predictions, targets ):
        # predictions: list of Value objects
        # targets :list pf Value objects (or numbers)

        n = len(predictions)
        loss=sum((p-t)**2 for p,t in zip(predictions, targets))
        return loss*(1.0/n) # Avearge

class SGD:
    """Stochastic Gradient Descent"""
    def __init__(self, parameters, lr=0.01):
        self.parameters=parameters
        self.lr=lr
    
    def step(self):
        for p in self.parameters:
            p.data -= self.lr *p.grad
    
    def zero_grad(self):
        for p in self.parameters:
            p.grad=0.0

    
    