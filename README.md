BLACK HOLE SHADOW SIMULATION / SPIN ESTIMATION

This project aims to calculate null geodesics in Kerr spacetimes of different spins by solving the Hamiltonian form of the geodesic equation numerically. By tracing multiple rays from the position of a stationary observer, images of the black hole shadow are created. As these images are just scatter plots that are computationally expensive, machine learning methods are used to increase resolution. From the produced high-resolution images, a CNN is trained to predict the value of the spin parameter from a given black hole shadow image. This is done only for a fixed observer position in the equatorial plane (r=50). The project is structured in the following way:

- src: contains files necessary for calculation and visualization of geodesics and shadows
    - calculations.py: calculates necessary values, such as metric, Christoffel symbols, etc. 
    - integration.py: implements the geodesic equation and integrates it using a Runge-Kutta method, paying attention to numerical error sources
    - visualization.py: visualizes the spatial part of the resulting geodesic as 3D embedding into euclidean R3
    - observer.py: calculates values necessary to describe the local frame of a stationary observer, such as local tetrad, etc. 
    - shadow_visualization.py: plots the shadow image as a scatter plot in p2 p3 phase space and allows to save initial conditions and the information, whether the event horizon was reached or not, in a csv file

- main.py: asks for user input to either visualize 3D geodesics, record geodesic data and save to a csv file, or visualize the black hole shadow for any spacetime parameters and initial conditions

- notebooks:
    - data_analysis.ipynb: data scientific analysis of the data collected for different initial conditions and spins, training of a neural network (MLP Classifier) that creates black hole shadow images (.png) for an observer in the equatorial plane (r=50) with high resolution
    - images: collection of images created by the MLP trained in data_analysis.ipynb for different spins a and a fixed observer position, the spin is recorded in the filename of the images
    
- data: directory to which the simulated data (initial conditions + information whether the event horizon was hit or not) can be saved and used for training of the image interpretation model


DISCLAIMERS:

- This project highlights one possible impact that machine learning can have on the visualization of physical phenomena. It does not reflect any physical reality or precision but rather a fun visualization technique that should be interpreted with caution.

- 
