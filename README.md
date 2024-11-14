# Non-Hermitian Density Matrix Renormalization Group

A library for density-matrix renormalization group for non-Hermitian systems. Still under development.

Modified based on the open-source (Hermitian) DMRG code on [tensors.net by Glen Evenbly](https://www.tensors.net/dmrg).
Non-Hermitian DMRG algorithm based on [1] and [2].

The code runs on Julia.Library requirements:
- ITensors
- Arpack
- MatrixEquations
- JLD2

[1] Zhong P, Pan W, Lin H, Wang X, Hu S. Density-matrix renormalization group algorithm for non-Hermitian systems. arXiv preprint arXiv:2401.15000.

[2] Yamamoto, K., Nakagawa, M., Tezuka, M., Ueda, M. and Kawakami, N., 2022. Universal properties of dissipative Tomonaga-Luttinger liquids: Case study of a non-Hermitian XXZ spin chain. Physical Review B, 105(20), p.205125.
