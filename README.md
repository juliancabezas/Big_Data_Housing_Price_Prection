# Big_Data_Housing_Price_Prection
Prediction of Price housing using tree-based algorithms

## Environment

This code was tested under a Linux 64 bit OS (Ubuntu 18.04 LTS), using Python 3.7

## How to run this code:

In order to use this repo:

1. Install Miniconda or Anaconda
2. Add conda forge to your list of channels

In the terminal run:
```
    conda config --add channels conda-forge
```
3. Create a environment using the requirements.yml file included in this repo, using the following command (inside conda or bash)

In the terminal run:
```
    conda env create -f requirements.yml --name house_regression
```
4. Activate the conda environment

In the terminal run:
```
    conda activate house_regression
```

5. Make sure the folder structure of the project is as follows

```
├── Intermediate_Results
├── Results
├── house_price_regression.py
├── README.txt
└── ...
```

If there are .csv files in the Intermediate_Results the code will read them to avoid the delay of the RFE and Gridsearch and go straigh to fitting the models

6. Run the house_price_regression.py file in your IDE of preference, (I recommend VS Code with the Python extension), using the root folder of the directory as working directory to make the relative paths work.


Note: Alternatevely, for 2 and 3 you can build your own environment following the package version contained in requirements.yml file

