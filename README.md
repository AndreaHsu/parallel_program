# Parallel Programming NTHU

Using `MPI`, `OpenMP`, `Pthread`, `CUDA` programming

### HW1: Odd-Even Sort
- [HW Spec](HW/hw1/PP_2023_HW1.pdf)
- [Report (Implementation Explanation & Experiments)](HW/hw1/hw1_109062101.pdf)
- [Implementation](HW/hw1/)
- Parallel Technique : Data Partition
- Programming Version: `MPI`
### HW2: Mandelbrot Set
- [HW Spec](HW/hw2/PP_2023_HW2.pdf)
- [Report (Implementation Explanation & Experiments)](HW/hw2/hw2_109062101.pdf)
- [Implementation](HW/hw2/)
- Parallel Technique: Data Partition, Vectorization
- Programming Version: `Pthread`, `MPI+OpenMP`
### HW3: All-Pair Shortest Path
- [HW Spec](HW/hw3/PP_2023_HW3.pdf)
- [Report (Implementation Explanation & Experiments)](HW/hw3/hw3_109062101.pdf)
- [Implementation](HW/hw3/)
- Parallel Technique: Data Partition, Shared Memory&Padding, Large Blocking Factor
- Programming Version: `CPU(OpenMP)`, `Single-GPU(CUDA)`, `Multi-GPU(CUDA)`
### HW4: UCX
- [HW Spec](HW/hw4/PP_2023_HW4.pdf)
- [Report (Implementation Explanation & Experiments)](HW/hw4/hw4_109062101.pdf)
- [Implementation](HW/hw4/hw4.diff) (Since [UCX project](https://github.com/NTHU-LSALAB/UCX-lsalab) is large, I only upload gitdiff to show my modification)