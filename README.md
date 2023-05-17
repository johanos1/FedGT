# FedGT

This repository includes the code for FedGT, a group-testing framework for identification of malicious clients in federated learning. 

## Usage
To run the code, create a conda environment named fedGT as
```bash
conda env create -f environment.yml
```

Next, to compile the C-code, navigate to src/C_code and run this from the terminal:
```bash
cc -fPIC -shared -o BCJR_4_python.so BCJR_4_python.c
```
to create BCJR_4_python.so that will be called for decoding in the defence/group_test.py class.

Create an empty folder called results, this is where the simulation will store the results as a txt file (JSON).

To configure the simulation, use the toml file in cfg_files and set the path in the beginning of fedGT_main.py. The different modes supported are: Oracle, no-defence, fedGT, and noiseless fedGT. Furthermore, two types of data poisoning are supported: untargeted (label permutations) and targeted (replace the labels of a given class).
The simulation is initiated by running 
```bash
python fedGT_main.py 
```
The image below is an example of a run for an untargeted attack.


![alt text](https://github.com/johanos1/TUM_Chalmers_AIswe/blob/anonymous_branch/example_img.png?raw=true)



## Credits
This repository is to some extent inspired from FedML (https://github.com/FedML-AI/FedML).
