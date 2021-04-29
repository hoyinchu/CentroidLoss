# A Loss Function to Incorporate Structural Information Contained In the Output Space

This is a project repository for the class DS 4420 Machine Learning and Data Mining 2

Project Paper title: A Loss Function to Incorporate Structural Information Contained In the Output Space
Project Paper Link: <insert later>

How can we teach our models to project onto a pre-defined output space? In
this paper we introduce centroid loss which allows models to incorporate output
space information in the forms of balls and radii. We demonstrate a use case
of this loss function by assigning a centroid and a radius to each node in an
ontology, projecting the ontology onto 2D space, then trained two models using
the centroid loss. We empirically show the loss function is effective by visualizing
the embedding produced by the models using this loss function. Finally, we
also evaluate the impact different output spaces would have on the accuracy of
the models using this loss.

## How to reproduce the result

Run the jupyter notebook CentroidLoss.ipynb blocks from top-down