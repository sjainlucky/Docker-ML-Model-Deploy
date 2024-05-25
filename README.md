# MNIST Digit Classification

This project demonstrates training and deploying a neural network to classify handwritten digits from the MNIST dataset. The deployment is containerized using Docker and exposed as a REST API using FastAPI. The application can be deployed on a Kubernetes cluster.

## Project Structure

mnist-inference/
├── app.py # FastAPI application script
├── Dockerfile # Dockerfile for building the container
├── inference.py # Inference script to predict digits
├── requirements.txt # Python dependencies
├── train.py # Training script for the model
├── deployment.yaml # Kubernetes deployment configuration
├── service.yaml # Kubernetes service configuration
└── README.md # Project documentation

## Steps to Run

1. docker build --platform=linux/arm64/v8 -t my-pytorch-app:latest .
2. docker run --name mycontainer -p 80:80 my-pytorch-app:latest

## Additional Information

### Model Architecture:

A simple neural network with one hidden layer.

### Dataset:

MNIST handwritten digits.

### Files Description

app.py: Implements the FastAPI server and the prediction endpoint.
inference.py: Contains the model definition and the prediction function.
train.py: Script to train the neural network on the MNIST dataset.
Dockerfile: Instructions to containerize the application.
deployment.yaml: Kubernetes deployment configuration.
service.yaml: Kubernetes service configuration.

## References

PyTorch
FastAPI
MNIST Dataset
