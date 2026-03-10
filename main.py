from aloygrad.tensor import Tensor

tensor_x=Tensor([[1,3,4],[2,4,5]])
tensor_y=Tensor([[1,3,6],[1,3,6],[1,3,6]])


print(tensor_x @ tensor_y)