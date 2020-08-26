# Big_Data_Housing_Price_Prection
Prediction of Price housing using tree-based algorithms

## Environment

This code was tested under a Linux 64 bit OS (Ubuntu 18.04 LTS), using Python 3.7.7

## How to run this code:

In order to use this repo:

1. Install Miniconda or Anaconda
2. Add conda forge to your list of channels

In the terminal run:

    conda config --add channels conda-forge

3. Create a environment using the requirements.yml file included in this repo:

Open a terminal in the folder were the requirements.yml file is (a1785086_Code_Project1) and run:

    conda env create -f requirements.yml --name house_regression


4. Make sure the folder structure of the project is as follows

a1785086_Code_Project1
├── Input_Data
├── Intermediate_Results
├── Results
├── house_price_regression.py
├── README.txt
└── ...

If there are .csv files in the Intermediate_Results the code will read them to avoid the delay of the RFE and Gridsearch and go straigh to fitting the models

5.  Run the code in the conda environment: Open a terminal in the a1785086_Code_Project1  and run 
	
	conda activate house_regression
	python house_price_regression.py


or run the house_price_regression.py code in your IDE of preference, (I recommend VS Code with the Python extension), using the root folder of the directory (a1785086_Code_Project1) as working directory to make the relative paths work.

Note: Alternatevely, for 2 and 3 you can build your own environment following the package version contained in requirements.yml file

