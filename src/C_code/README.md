# TUM_Chalmers_AIswe

To create the shared library to use C functions in Python, run this from the terminal:

cc -fPIC -shared -o my_functions.so my_functions.c
```
all the code in my_functions.c will be available by calling
```python
my_functions = CDLL(so_file)
```
in the Python module.
