<img src="/imgs/graphical_model.png" width="100%" height="100%">

# Forecasting Real-time Traffic Information Based on Historical Traffic Data
This repository is dedicated to the results of the DS535 class in the fall of 2023. We utilized GNN for traffic forecasting while simultaneously incorporating hierarchical information of administrative areas to propose a novel model. A summarized meterial of the project can be found in the [G4_DS535_final_presentation](https://github.com/GwangWooKim/RecSys-GNN/blob/main/G4_DS535_final_presentation.pdf). The final report is available [here](https://www.overleaf.com/read/jzbyzqgfywbb#cdb988).

## Setup for reproducing our results
* python == 3.10.12
* pytorch == 2.1.1
* torch_geometric == 2.4.0
* torch-geometric-temporal == 0.54.0
* matplotlib
* pyarrow

In order to reproduce our results, create a new environment with python == 3.10.12 and run `setup.sh`. For example, 
```
$ sh setup.sh
```
**Note that the shall script is running on cuda 12.1, so modify the `setup.sh` file up to your local machine.**
