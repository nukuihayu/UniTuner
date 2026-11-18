"""Adapter operations with native fallbacks and lazy CUDA kernels.

Algorithm modules own dispatch and autograd; matching *_cuda modules own
Triton kernels. Keep this package free of eager backend imports.
"""
