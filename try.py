import os

""" import torch
print(torch.__version__)
print(torch.cuda.is_available()) """

""" program_directory = "E:\Documents\GitHub\Week3Try"
os.chdir(program_directory)

current_working_directory = os.getcwd()
print(current_working_directory) """

import torch

# Create a 1-dimensional tensor of size 4
vec = torch.tensor([1.0, 2.0, 3.0, 4.0])
print("Vector:", vec)

# Create a 2-dimensional tensor (matrix) of size 2x3
mat = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("Matrix:\n", mat)

# Create a 3-dimensional tensor of size 2x2x3
tensor_3d = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print("3D Tensor:\n", tensor_3d)

names = ["alice", "bob", "charlie"]
upper_names = [name.upper() for name in names]
print(upper_names)  # Output: ['ALICE', 'BOB', 'CHARLIE']
