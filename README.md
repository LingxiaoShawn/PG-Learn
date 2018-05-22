# PG-Learn
An efficient and effective algorithm of learning graph for semi-supervised learning. (MATLAB Code)  

## Instruction: Run code & examples
Before use the code you should compile mtimesx lib, which is inside util/lib/mtimesx/ folder. Please 
refer to [mtimesx](https://cn.mathworks.com/matlabcentral/fileexchange/25977-mtimesx-fast-matrix-multiply-with-multi-dimensional-supporthttps://cn.mathworks.com/matlabcentral/fileexchange/25977-mtimesx-fast-matrix-multiply-with-multi-dimensional-support). For Mac OS users, you can first use **Homebrew** to install **openblas** library, and then run
 
 	bias_lib = 'path to libblas.dylib'
	mex('-DDEFINEUNIX','-largeArrayDims','mtimesx.c',blas_lib)
	
After install required library, you should excute **main.m** in the root folder. After that you can run all matlab files under root folder. 
	
In the **example** folder, we provide examples with respect to single-thread version PG-Learn, hyperband-parallel version PG-Learn, and several baselines including grid search, random search, MinEnt, AEW and IDML. What's more, we also provide example of running the relatively general parallel framework **RndSet\_parallel_framework**.

Notice that the codes of baselines are kind of messy. These codes are designed for personal usage. 

## Data 
There are six wide-use image datasets under **datasets** folder. These datasets are benchmark datasets used in our paper. Under each specific benckmark dataset folder, the **DatasetName.mat** file is the original data. All other ***PerTrain** subfolders are used specifically by **LoadData** function. The **LoadData** function is a personal used function to transform the original data into the **data** required by algorithms.
 

## Cite
Please cite our paper if you use our code in your research. 
