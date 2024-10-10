# DRLDetectionFSM
Coupling an FSM with DRL to improve interpretability and convergence

# Installation
The virtual environment packages used are described in the "requirements.txt" file. It can be used with :
pip -r requirements.txt

# Training 
In order to launch a training with a specific method, you have to specify the hyperparameters ins the config.yaml file and to launch the script. Training will be launched accordingly to the parameters referenced with the automatic save of checkpoints of the model and summary plots.   
The "Changing parameters" section permits to launch multiple trainings sequentially while potentially changing hyperparameters.

# Inference
Once trainings are done through a training script, a parallel evaluation of the checkpoint on multiple episodes can be performed thanks to "analysys/inferenceNN/inferenceD3QN.py." if you reference the path to the folder containing the multiples trainings saves.

# Clustering
In order to use "analysis/clustering/clustering.py" script, you have to reference the folder of the trainings on wich you create the centroids and the list of foler of trainings of which you apply the clustering with those centroids.

# Plots
The "analysis/plots" scripts permits to create multiple plots :
- The average zeta values over multiple trainings.
- The comparison of inference values with 95% confidence interval depending on the method used
- The comparison of inference values with 95% confidence interval depending on the cluster it was assigned
- The violin plot of the average timestep spent in each state over the whole training depending on the method used

# Sources
The initial algorithm on which this approach is based is from :
Z. Wang, T. Schaul, M. Hessel, H. van Hasselt,M. Lanctot and N. de Freitas, “ Dueling NetworkArchitectures for Deep Reinforcement Learning, ”in JMLR, 2016.

The main framework is TorchRL and from which the base implementation of D3QN was taken :
A. Bou, M. Bettini, S. Dittert et al. “ TorchRL : Adata-driven decision-making library for PyTorch.” arXiv : 2306.00577.

Due to the size of the data generated for the article plot, it wasn't possible to add the source data. 
If there is any questions or feedback, feel free to submit it.
