# FedGT - Compilation of C Code

To create the shared library to use C functions in Python, run this from the terminal:

for FedGT decoder: 

cc -fPIC -shared -o BCJR_4_python.so BCJR_4_python.c

and for FedQGT decoder:

cc -fPIC -shared -o ./QGT/FedQGT_decoder.so FedQGT_decoder.c

```
all the code in my_functions.c will be available by calling
```python
my_functions = CDLL(so_file)
```
in the Python module.
