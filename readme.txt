### Yu Shuai's source code for MAE557 Assignment 1

### 1. Prerequisities (tested on Adroit)




language: c++ (c++17)
compiler: g++ (GCC) 11.5.0 from GNU
library: eigen3 3.4.0




### 2. Compiling and running




in Adroit terminal, type the following for compilation:

g++ -std=c++17 -I/usr/include/eigen3 your_name.cpp -o your_name

then type the following for running the executable:

./your_name



Here, your_name can be solver_explicit or solver_implicit




### 3. Postprocessing




After running the codes, there will be .txt files generated named such as:

burgers_explicit_nu_0.01_nx_120_dt_0.001.txt

These files can then be collected for plotting figures using the Python program plot.py


