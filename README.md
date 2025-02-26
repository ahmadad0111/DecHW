# This repository contains the implementation of DecHW in the SAISIM simulator.
# SaiSim
Simulator for SAI project

## Project structure

### Directories
**training_strategies/** contains the learning schemes for models' aggregation and local model training

**graph/** contains the graph generators

**connectivity/** (unused)

**data**/ it will contain the data for training models (data are downloaded automatically)

**utils/** containes all the utilities for the project

**stats/** contains the output log/statistics

**training_style/** contains the code for different style of trainig (decentralise, centralised, federated)

### Files (root)
**clock.py** contains the simulator clock 

**main.py** is the main for running the simulator

**message.py** contains the definition and management of the messaging between PAIVS

**models.py** contains DNN models' definitions

**paiv.py** contains the definion of the PAIV(s)

**run_saisim.sh** is a shell script to run the main.py with input arguments 
