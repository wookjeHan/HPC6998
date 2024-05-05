import torch

MAGIC_NUMBER=15

A = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])  # Example tensor

ADDITIONAL_ATTENTION_MASK = [[-float('inf') for i in range(MAGIC_NUMBER)] for j in range(MAGIC_NUMBER)]
for i in range(MAGIC_NUMBER):
    ADDITIONAL_ATTENTION_MASK[i][i] = 0
    ADDITIONAL_ATTENTION_MASK[i][0] = 0
    for j in range(1, i):
        cur = i
        while cur != 0:
            cur = (cur-1)//2
            ADDITIONAL_ATTENTION_MASK[i][cur] = 0
ADDITIONAL_ATTENTION_MASK = torch.tensor(ADDITIONAL_ATTENTION_MASK)
print("B")
print(ADDITIONAL_ATTENTION_MASK)
print("A+B")
print(A+ADDITIONAL_ATTENTION_MASK)
IS_ZERO = (A+ADDITIONAL_ATTENTION_MASK != 0).all(dim=1)
print("ISNOTZERO")
print(IS_ZERO)
