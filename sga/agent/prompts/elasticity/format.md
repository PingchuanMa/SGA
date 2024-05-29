### PyTorch Tips
1. When element-wise multiplying two matrix, make sure their number of dimensions match before the operation. For example, when multiplying `J` (B,) and `I` (B, 3, 3), you should do `J.view(-1, 1, 1)` before the operation. Similarly, `(J - 1)` should also be reshaped to `(J - 1).view(-1, 1, 1)`. If you are not sure, write down every component in the expression one by one and annotate its dimension in the comment for verification.
2. When computing the trace of a tensor A (B, 3, 3), use `A.diagonal(dim1=1, dim2=2).sum(dim=1).view(-1, 1, 1)`. Avoid using `torch.trace` or `Tensor.trace` since they only support 2D matrix.

### Code Requirements

1. The programming language is always python.
2. Annotate the size of the tensor as comment after each tensor operation. For example, `# (B, 3, 3)`.
3. The only library allowed is PyTorch. Follow the examples provided by the user and check the PyTorch documentation to learn how to use PyTorch.
4. Separate the code into continuous physical parameters that can be tuned with differentiable optimization and the symbolic constitutive law represented by PyTorch code. Define them respectively in the `__init__` function and the `forward` function.
5. The first output of the `forward` function is the updated deformation gradient. Always remember the second output of the `forward` function is Kirchhoff stress tensor, which is defined by the matrix multiplication between the first Piola-Kirchhoff stress tensor and the transpose of the deformation gradient tensor. Formally, `tau = P @ F^T`, where tau is the Kirchhoff stress tensor, P is the first Piola-Kirchhoff stress tensor, and F is the deformation gradient tensor. Do not directly return any other type of stress tensor other than Kirchhoff stress tensor. Compute Kirchhoff stress tensor using the equation: `tau = P @ F^T`.
6. The proposed code should strictly follow the structure and function signatures below:

```python
{code}
```

### Solution Requirements

1. Analyze step-by-step what the potential problem is in the previous iterations based on the feedback. Think about why the results from previous constitutive laws mismatched with the ground truth. Do not give advice about how to optimize. Focus on the formulation of the constitutive law. Start this section with "### Analysis". Analyze all iterations individually, and start the subsection for each iteration with "#### Iteration N", where N stands for the index. Remember to analyze every iteration in the history.
2. Think step-by-step what you need to do in this iteration. Think about how to separate your algorithm into a continuous physical parameter part and a symbolic constitutive law part. Describe your plan in pseudo-code, written out in great detail. Remember to update the default values of the trainable physical parameters based on previous optimizations. Start this section with "### Step-by-Step Plan".
3. Output the code in a single code block "```python ... ```" with detailed comments in the code block. Do not add any trailing comments before or after the code block. Start this section with "### Code".