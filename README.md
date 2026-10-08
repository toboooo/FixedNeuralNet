"From scratch" C++ code to solve the "I dropped a neural net" puzzle by Jane Street: https://huggingface.co/spaces/jane-street/droppedaneuralnet.

Implements a solution by Hyunwoo Park: https://arxiv.org/abs/2602.19845. The steps are as follows:

1. Greedy pairing of the "input" and "output" weights, $W^{in}$ and $W^{out}$, by the maximum of the ratio between the trace and the Frobenius norm of their products $W^{out}W^{in}$.

2. Sorting the weight pairs by increasing Frobenius norm of the output weight.

3. Hill-climbing on MSE by greedily swapping adjacent weight pairs until no more swaps lead to an improvement.

The `make_dat_files.py` script works on a subdirectory called `pieces` that contains all of the PyTorch `.pth` files from the problem site. It creates `.dat` files that `fix_network` needs to work. The `pieces` folder and `historical_data.csv` should be placed into the same working directory as the program's execution. Compile using:

```
gcc -Ofast -ffast-math -funroll-loops -march=native fix_network.cpp network.cpp data_matrix.cpp -o fix_network -lstdc++ -lm
```

The matrix multiplication is slow and could be improved.
