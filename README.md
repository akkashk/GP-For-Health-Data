# Using Gaussian Process to model health metrics from fitnedd trackers

Working with Nokia Health as an external client, I developed a framework for using Gaussian Processes to model health metrics obtained from fitness trackers (e.g. steps taken/hours slept) and using the fitted hyperparameters to embed each user in the hyperparameter space to find clusters of users that share characteristics. I used Silhouette analysis to find the ideal number of clusters and used aggregation techniques to fit better Gaussian Process models with higher marginal likelihood values
