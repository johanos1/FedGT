# FedGT
​
This repository includes the code for FedGT, a group-testing framework for identification of malicious clients in federated learning. 
​
## Usage
To run the code, create a conda environment named fedGT as
```bash
conda env create -f environment.yml
```
then activate the environment as
```bash
conda activate fedGT
```

Create an empty folder called results, this is where the simulation will store the results as a txt file (JSON).
To configure the simulation, use the toml file in cfg_files and set the path in the beginning of fedGT_main.py. Datasets supported are MNIST, CIFAR10. 
You may run the federated learning by the following command:
```bash
python3 main_threaded.py
```
​

​
​
## Credits
This repository is to some extent inspired from FedML (https://github.com/FedML-AI/FedML). Some functions are taken from FLamby (https://github.com/owkin/FLamby).
