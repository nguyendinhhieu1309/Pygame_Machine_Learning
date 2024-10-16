# KNN and KMeans Visualization with Pygame
![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Pygame](https://img.shields.io/badge/Library-Pygame-green?style=for-the-badge&logo=pygame)
![Numpy](https://img.shields.io/badge/Library-Numpy-013243?style=for-the-badge&logo=numpy)
![Scikit-learn](https://img.shields.io/badge/Library-Scikit--learn-orange?style=for-the-badge&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge)

This repository provides interactive visualizations of K-Nearest Neighbors (KNN) and KMeans clustering algorithms implemented using Python's `pygame` library. The tool enables users to explore how both algorithms work by visualizing the clustering and classification processes in real-time, along with linear regression visualizations.

## Features

- **K-Nearest Neighbors (KNN):**
  - Visualize how KNN works by selecting points on a 2D plane.
  - Adjust the value of `K` (number of neighbors).
  - Run KNN classification on newly added points to see real-time results.
  
- **KMeans Clustering:**
  - Dynamically create clusters by adding random points or clicking to add custom points.
  - Adjust the number of clusters (`K`).
  - Run the KMeans algorithm to see how points are classified into clusters.
  - View the movement of cluster centroids after each iteration.

- **Linear Regression:**
  - Add data points to the 2D plane and fit a linear regression line.
  - Visualize the result of regression both through a custom formula and using the scikit-learn library.
  - Clear the regression lines or reset the data at any point.
  - Display error values for regression models.

## Getting Started

### Prerequisites

To run this project, you will need the following dependencies:

- Python 3.x
- `pygame`
- `numpy`
- `scikit-learn`

You can install the required libraries using pip:

```bash
pip install pygame numpy scikit-learn
