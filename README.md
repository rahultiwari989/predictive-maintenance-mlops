# Predictive Maintenance MLOps

An end-to-end **Predictive Maintenance and Remaining Useful Life (RUL) prediction system** built using deep learning, FastAPI, Docker, Kubernetes, Google Cloud Platform, GKE Autopilot, Artifact Registry, Streamlit, and GitHub Actions.

The project demonstrates how a machine-learning model can move from local development to a reproducible, containerized, continuously deployed production-style environment.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Business Problem](#business-problem)
* [Solution](#solution)
* [Architecture](#architecture)
* [Technology Stack](#technology-stack)
* [Project Structure](#project-structure)
* [Machine Learning Pipeline](#machine-learning-pipeline)
* [Model Serving](#model-serving)
* [Streamlit Dashboard](#streamlit-dashboard)
* [Docker](#docker)
* [Kubernetes Architecture](#kubernetes-architecture)
* [Google Cloud Deployment](#google-cloud-deployment)
* [Artifact Registry](#artifact-registry)
* [CI/CD Pipeline](#cicd-pipeline)
* [Workload Identity Federation](#workload-identity-federation)
* [Deployment Strategy](#deployment-strategy)
* [Rollback Strategy](#rollback-strategy)
* [Health and Readiness Checks](#health-and-readiness-checks)
* [Horizontal Pod Autoscaling](#horizontal-pod-autoscaling)
* [Local Kubernetes Deployment](#local-kubernetes-deployment)
* [Production Deployment](#production-deployment)
* [Testing](#testing)
* [End-to-End Validation](#end-to-end-validation)
* [Troubleshooting](#troubleshooting)
* [MLOps Design Decisions](#mlops-design-decisions)
* [Production Improvements](#production-improvements)
* [Interview Explanation](#interview-explanation)

---

# Project Overview

This project implements a production-style predictive maintenance system that predicts the **Remaining Useful Life (RUL)** of an aircraft engine from multivariate time-series sensor data.

The system uses historical sensor measurements from the NASA C-MAPSS FD001 dataset.

The trained deep-learning model receives a time-series window of sensor observations and predicts the estimated remaining useful life of the engine.

The complete application consists of:

```text
Machine Learning Model
        ↓
FastAPI Inference API
        ↓
Docker Container
        ↓
Kubernetes
        ↓
GKE Autopilot
        ↓
Streamlit Dashboard
```

The project also implements automated CI/CD:

```text
Git Push
   ↓
GitHub Actions
   ↓
Unit Tests
   ↓
Docker Build
   ↓
Artifact Registry
   ↓
GKE Deployment
   ↓
Rolling Update
   ↓
Production Inference
```

---

# Business Problem

Predictive maintenance aims to identify potential equipment degradation before an actual failure occurs.

Traditional maintenance approaches often rely on:

* Fixed maintenance schedules
* Manual inspections
* Reactive maintenance after failure

These approaches can result in:

* Unexpected equipment downtime
* Unnecessary maintenance
* Increased operational costs
* Reduced equipment availability

A predictive maintenance system instead uses historical sensor data to estimate equipment health and remaining useful life.

The objective of this project is therefore:

> **Given a sequence of aircraft engine sensor observations, predict the remaining useful life of the engine.**

---

# Solution

The system uses a deep-learning time-series model to learn degradation patterns from historical engine sensor measurements.

The inference flow is:

```text
Sensor Data
     ↓
Preprocessing
     ↓
Feature Selection
     ↓
Scaling
     ↓
30-Time-Step Window
     ↓
CNN-LSTM Model
     ↓
RUL Prediction
```

The deployed application exposes the prediction through FastAPI and provides a Streamlit dashboard for visualization and inference.

---

# Architecture

## High-Level Architecture

```text
                         ┌──────────────────────┐
                         │      Developer       │
                         └──────────┬───────────┘
                                    │
                                 git push
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       GitHub         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   GitHub Actions     │
                         │      CI / CD         │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    │                                │
                    ▼                                ▼
              pytest tests                    Docker Build
                                                     │
                                                     ▼
                                            Artifact Registry
                                                     │
                                                     ▼
                                             GKE Autopilot
                                                     │
                          ┌──────────────────────────┴────────────────────┐
                          │                                               │
                          ▼                                               ▼
                 Streamlit Dashboard                               FastAPI Service
                    LoadBalancer                                      ClusterIP
                          │                                               │
                          │                                               ▼
                          │                                      ┌─────────────────┐
                          │                                      │   API Pod 1     │
                          │                                      │ CNN-LSTM Model  │
                          │                                      └─────────────────┘
                          │                                               │
                          │                                      ┌─────────────────┐
                          └─────────────────────────────────────►│   API Pod 2     │
                                                                 │ CNN-LSTM Model  │
                                                                 └─────────────────┘
                                                                          │
                                                                          ▼
                                                                    RUL Prediction
```

---

# Technology Stack

| Layer                | Technology                   |
| -------------------- | ---------------------------- |
| Programming Language | Python                       |
| Machine Learning     | TensorFlow / Keras           |
| Model                | CNN-LSTM                     |
| API                  | FastAPI                      |
| Dashboard            | Streamlit                    |
| Containerization     | Docker                       |
| Local Orchestration  | Kubernetes / Minikube        |
| Cloud Orchestration  | Google Kubernetes Engine     |
| GKE Mode             | Autopilot                    |
| Container Registry   | Google Artifact Registry     |
| CI/CD                | GitHub Actions               |
| Cloud Authentication | Workload Identity Federation |
| Testing              | pytest                       |
| API Communication    | HTTP                         |
| Scaling              | Kubernetes HPA               |
| Source Control       | Git / GitHub                 |

---

# Project Structure

```text
predictive-maintenance-mlops/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── api/
│   └── main.py
│
├── artifacts/
│   ├── cnn_lstm_model.keras
│   ├── metadata.json
│   └── scaler.pkl
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── RUL_FD001.txt
│   ├── test_FD001.txt
│   └── train_FD001.txt
│
├── kubernetes/
│   ├── api-deployment.yaml
│   ├── api-service.yaml
│   ├── dashboard-deployment.yaml
│   ├── dashboard-service.yaml
│   └── hpa.yaml
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
│
├── tests/
│   ├── __init__.py
│   ├── e2e_api.py
│   ├── load_generator.py
│   └── test_unit.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── Dockerfile.dashboard
├── docker-compose.yml
├── README.md
├── requirements.txt
└── requirements-dashboard.txt
```

---

# Machine Learning Pipeline

The machine-learning pipeline uses multivariate time-series sensor data.

The input contains:

```text
Aircraft Engine
      ↓
Multiple sensor measurements
      ↓
Time-series sequence
      ↓
Feature preprocessing
      ↓
Sliding window
      ↓
CNN-LSTM
      ↓
RUL
```

## Input Window

The deployed inference model uses:

```text
Window size: 30 time steps
Features:    16
```

Therefore an inference window has the shape:

```text
(30, 16)
```

The model produces a scalar RUL prediction.

---

# Model Serving

The trained model is stored as:

```text
artifacts/cnn_lstm_model.keras
```

The scaler is stored as:

```text
artifacts/scaler.pkl
```

Metadata is stored as:

```text
artifacts/metadata.json
```

The FastAPI application loads these artifacts during application startup.

Conceptually:

```python
predictor = RULPredictor()
```

which loads the trained model and preprocessing artifacts.

---

# FastAPI Endpoints

The inference API provides health and prediction functionality.

## Health

```text
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Readiness

```text
GET /ready
```

This endpoint is used by Kubernetes to determine whether the API pod is ready to receive traffic.

## Prediction

The API accepts a raw inference window and returns an RUL prediction.

Example result:

```json
{
  "predicted_rul": 96.45166015625,
  "unit": "cycles",
  "status": "healthy"
}
```

---

# Streamlit Dashboard

The Streamlit dashboard provides a user interface for interacting with the predictive maintenance system.

The dashboard:

1. Loads the FD001 test data.
2. Allows the user to select an engine/inference window.
3. Sends the sensor window to FastAPI.
4. Receives the predicted RUL.
5. Displays the predicted equipment health.

The dashboard communicates with FastAPI using the Kubernetes service:

```text
http://predictive-maintenance-api:8000
```

This allows the API to remain internal to the Kubernetes cluster.

---

# Docker

Two Docker images are used.

## API Image

```text
predictive-maintenance-api
```

Built using:

```bash
docker build -t predictive-maintenance-api:1.0 .
```

## Dashboard Image

```text
predictive-maintenance-dashboard
```

Built using:

```bash
docker build \
  -f Dockerfile.dashboard \
  -t predictive-maintenance-dashboard:1.0 \
  .
```

---

# Kubernetes Architecture

The application is deployed using Kubernetes Deployments and Services.

## API Deployment

The API uses:

```text
Replicas: 2
Port: 8000
```

This provides multiple inference pods.

```text
API Deployment
      │
 ┌────┴────┐
 ▼         ▼
Pod 1     Pod 2
```

## API Service

The API uses:

```yaml
type: ClusterIP
```

The service is:

```text
predictive-maintenance-api
```

The API is therefore accessible internally through:

```text
http://predictive-maintenance-api:8000
```

The API is not directly exposed to the public internet.

---

# Dashboard Service

The Streamlit dashboard is exposed externally through a Kubernetes LoadBalancer.

```yaml
type: LoadBalancer
```

The dashboard listens on:

```text
8501
```

Example:

```text
http://<EXTERNAL-IP>:8501
```

---

# Google Cloud Deployment

The project is deployed to Google Cloud using GKE Autopilot.

## GCP Configuration

```text
Project:
pm-mlops-gcp-2026

Region:
asia-south1

GKE Cluster:
predictive-maintenance

Artifact Registry:
predictive-maintenance
```

---

# Authenticate with Google Cloud

```bash
gcloud auth login
```

Set the project:

```bash
gcloud config set project pm-mlops-gcp-2026
```

Verify:

```bash
gcloud config get-value project
```

---

# Connect kubectl to GKE

```bash
gcloud container clusters get-credentials predictive-maintenance \
  --region asia-south1 \
  --project pm-mlops-gcp-2026
```

Verify the cluster:

```bash
kubectl get nodes
```

Expected:

```text
NAME                                      STATUS   ROLES
gk3-predictive-maintenance-...            Ready    <none>
```

---

# Artifact Registry

Docker images are stored in Google Artifact Registry.

The registry path is:

```text
asia-south1-docker.pkg.dev/pm-mlops-gcp-2026/predictive-maintenance
```

The API image follows:

```text
asia-south1-docker.pkg.dev/pm-mlops-gcp-2026/predictive-maintenance/predictive-maintenance-api:<git-sha>
```

The dashboard image follows:

```text
asia-south1-docker.pkg.dev/pm-mlops-gcp-2026/predictive-maintenance/predictive-maintenance-dashboard:<git-sha>
```

Images are versioned using Git commit SHA values.

This avoids the ambiguity associated with using a mutable `latest` tag.

---

# CI/CD Pipeline

GitHub Actions provides the CI/CD pipeline.

The pipeline follows:

```text
Developer
    │
    ▼
git push
    │
    ▼
GitHub
    │
    ▼
GitHub Actions
    │
    ▼
Run Tests
    │
    ├── Failure → Stop
    │
    ▼
Build Docker Images
    │
    ▼
Push to Artifact Registry
    │
    ▼
Connect to GKE
    │
    ▼
Update Kubernetes Deployment
    │
    ▼
Rolling Update
    │
    ▼
Verify Rollout
```

---

# Continuous Integration

For pull requests and pushes to `main`, the test stage runs:

```bash
python -m pytest -v
```

The current unit test suite validates:

* Window configuration
* Feature configuration
* RUL cap configuration
* Required model artifacts

Example successful result:

```text
4 passed
```

---

# Continuous Deployment

For pushes to `main`, the pipeline:

1. Runs tests.
2. Builds the API image.
3. Builds the dashboard image.
4. Authenticates to GCP.
5. Pushes both images to Artifact Registry.
6. Connects to GKE.
7. Updates the Kubernetes deployments.
8. Waits for the rollout to complete.

The deployment uses Git SHA-based images.

Example:

```text
predictive-maintenance-api:
9f65c4514e13dc85f6963dc578c68073ead7ef32
```

This creates a direct relationship between:

```text
Git Commit
    ↓
Docker Image
    ↓
Artifact Registry
    ↓
GKE Deployment
```

---

# Workload Identity Federation

GitHub Actions authenticates to Google Cloud using **Workload Identity Federation**.

The pipeline does not depend on storing a long-lived GCP service-account JSON key inside GitHub.

The architecture is:

```text
GitHub Actions
      │
      │ OIDC Token
      ▼
Google Workload Identity Federation
      │
      ▼
GCP Service Account
      │
      ├── Artifact Registry permissions
      └── GKE deployment permissions
```

The federated identity is restricted to the GitHub repository:

```text
rahultiwari989/predictive-maintenance-mlops
```

This improves security and reduces the risk associated with long-lived cloud credentials.

---

# Deployment Strategy

The API deployment uses Kubernetes `RollingUpdate`.

The production deployment has two API replicas.

Example:

```text
Version A

Pod 1 ── Running
Pod 2 ── Running
```

When Version B is deployed:

```text
Version A
Pod 1 ── Running
Pod 2 ── Running

Version B
Pod 3 ── Starting
```

Once the new pod becomes ready:

```text
Version A
Pod 1 ── Running

Version B
Pod 3 ── Ready
```

Kubernetes can then progressively replace the old version.

This allows application updates without requiring all replicas to be unavailable simultaneously.

---

# Health and Readiness Checks

The API uses two Kubernetes probes.

## Readiness Probe

```text
GET /ready
```

The readiness probe determines whether a pod is ready to receive traffic.

Configuration:

```text
Initial delay: 10 seconds
Period:        5 seconds
```

The model must successfully load before the pod becomes ready.

This is especially important for ML serving applications because model loading can take time.

---

# Liveness Probe

```text
GET /health
```

The liveness probe determines whether the container is still functioning.

Configuration:

```text
Initial delay: 20 seconds
Period:        10 seconds
```

If the container becomes unhealthy, Kubernetes can restart it.

---

# Rollback Strategy

Kubernetes maintains deployment revisions.

View deployment history:

```bash
kubectl rollout history deployment/predictive-maintenance-api
```

Check the current rollout:

```bash
kubectl rollout status deployment/predictive-maintenance-api
```

If a deployment fails:

```bash
kubectl rollout status deployment/predictive-maintenance-api
```

The previous version can be restored using:

```bash
kubectl rollout undo deployment/predictive-maintenance-api
```

Verify:

```bash
kubectl rollout status deployment/predictive-maintenance-api
```

Expected:

```text
deployment "predictive-maintenance-api" successfully rolled out
```

---

# Failed Deployment Protection

If a new API container crashes:

```text
OLD VERSION
├── Pod 1   Running
└── Pod 2   Running

NEW VERSION
└── Pod     CrashLoopBackOff
```

The existing healthy replicas can continue serving traffic while Kubernetes attempts the rolling update.

This behavior was validated during deployment.

A failed image caused the new API pod to enter:

```text
CrashLoopBackOff
```

while the previous two API replicas remained healthy.

After rollback, the API returned to:

```text
2/2 Running
```

---

# Horizontal Pod Autoscaling

The API deployment is configured with a Horizontal Pod Autoscaler.

Example configuration:

```text
Minimum replicas: 2
Maximum replicas: 5
CPU target:       60%
```

The purpose is to allow the inference service to scale horizontally when CPU utilization increases.

Check HPA:

```bash
kubectl get hpa
```

Example:

```text
NAME                              TARGETS
predictive-maintenance-api-hpa   cpu: 1%/60%
```

The minimum replica count remains two to provide baseline availability.

---

# Local Kubernetes Deployment

The application was initially validated using Kubernetes locally.

Minikube was used to validate:

* Kubernetes Deployments
* Services
* API-to-dashboard communication
* Health checks
* HPA behavior
* Container orchestration

Example:

```bash
minikube start
```

Check:

```bash
kubectl get nodes
```

Deploy the API:

```bash
kubectl apply -f kubernetes/api-deployment.yaml
kubectl apply -f kubernetes/api-service.yaml
```

Deploy the dashboard:

```bash
kubectl apply -f kubernetes/dashboard-deployment.yaml
kubectl apply -f kubernetes/dashboard-service.yaml
```

---

# Production Deployment

The same Kubernetes concepts were then deployed to GKE Autopilot.

## Apply API Deployment

```bash
kubectl apply -f kubernetes/api-deployment.yaml
```

## Apply API Service

```bash
kubectl apply -f kubernetes/api-service.yaml
```

## Apply Dashboard Deployment

```bash
kubectl apply -f kubernetes/dashboard-deployment.yaml
```

## Apply Dashboard Service

```bash
kubectl apply -f kubernetes/dashboard-service.yaml
```

## Apply HPA

```bash
kubectl apply -f kubernetes/hpa.yaml
```

---

# Verify Kubernetes Resources

Check pods:

```bash
kubectl get pods
```

Check deployments:

```bash
kubectl get deployments
```

Check services:

```bash
kubectl get services
```

Check HPA:

```bash
kubectl get hpa
```

Check rollout:

```bash
kubectl rollout status deployment/predictive-maintenance-api
```

Check dashboard rollout:

```bash
kubectl rollout status deployment/predictive-maintenance-dashboard
```

---

# Verify Internal API Communication

The API is intentionally exposed only as a ClusterIP service.

Test it from inside Kubernetes:

```bash
kubectl run api-test \
  --rm \
  -it \
  --image=curlimages/curl \
  --restart=Never \
  -- curl http://predictive-maintenance-api:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

This confirms Kubernetes service discovery and internal API communication.

---

# End-to-End Validation

The final cloud deployment was validated through the Streamlit dashboard.

The inference window used:

```text
Shape: 30 × 16
```

The complete inference flow was:

```text
FD001 Test Data
      ↓
Streamlit Dashboard
      ↓
GKE LoadBalancer
      ↓
Streamlit Container
      ↓
Kubernetes ClusterIP
      ↓
FastAPI
      ↓
Preprocessing / Scaling
      ↓
CNN-LSTM Model
      ↓
RUL Prediction
```

The deployed system produced:

```text
Predicted RUL: 96.45 cycles
Status: healthy
```

This confirms that the model, API, dashboard, Docker images, Kubernetes services, GKE deployment, and runtime artifacts are working together.

---

# Runtime Artifacts

The Docker image requires the following runtime files:

```text
artifacts/
├── cnn_lstm_model.keras
├── scaler.pkl
└── metadata.json
```

The dashboard requires:

```text
data/
└── test_FD001.txt
```

These files are included in the repository so that the CI/CD Docker build is reproducible.

A key deployment lesson from this project was that files available locally are not automatically available to GitHub Actions.

All required runtime artifacts must therefore be:

* committed to source control,
* downloaded during the build,
* or retrieved from a model/data registry or object store.

---

# Testing

The project uses `pytest`.

Run:

```bash
python -m pytest -v
```

Example:

```text
tests/test_unit.py::test_window_configuration PASSED
tests/test_unit.py::test_feature_configuration PASSED
tests/test_unit.py::test_rul_cap_is_valid PASSED
tests/test_unit.py::test_model_artifacts_exist PASSED

4 passed
```

---

# Troubleshooting

## `ModuleNotFoundError: No module named 'src'`

Make sure the project root is the working directory:

```bash
cd predictive-maintenance-mlops
```

The repository contains:

```text
src/
├── __init__.py
├── config.py
├── model.py
├── predict.py
├── preprocessing.py
└── train.py
```

Run tests from the project root:

```bash
python -m pytest -v
```

---

## Model File Not Found

If the API reports:

```text
File not found:
/app/artifacts/cnn_lstm_model.keras
```

verify:

```bash
git ls-files artifacts
```

The model must be tracked by Git and included in the Docker build context.

---

## Test Data Not Found

If Streamlit reports:

```text
No such file or directory:
/app/data/test_FD001.txt
```

verify:

```bash
git ls-files data
```

The required dataset must be available to the Docker build.

---

## CrashLoopBackOff

Check:

```bash
kubectl get pods
```

Then:

```bash
kubectl logs <pod-name>
```

For a previous crashed container:

```bash
kubectl logs <pod-name> --previous
```

Inspect the pod:

```bash
kubectl describe pod <pod-name>
```

---

## Deployment Exceeded Progress Deadline

Check:

```bash
kubectl rollout status deployment/predictive-maintenance-api
```

Then:

```bash
kubectl get pods
```

and:

```bash
kubectl describe deployment predictive-maintenance-api
```

Common causes include:

* Container crash
* Missing model artifact
* Missing data
* Failed readiness probe
* Image pull failure
* Resource constraints

---

## GKE Authentication Error

If `kubectl` reports:

```text
gke-gcloud-auth-plugin.exe not found
```

install the GKE authentication plugin and reconnect to the cluster.

Then run:

```bash
gcloud container clusters get-credentials predictive-maintenance \
  --region asia-south1 \
  --project pm-mlops-gcp-2026
```

---

# MLOps Design Decisions

## Why FastAPI?

FastAPI provides:

* Lightweight model serving
* REST-based inference
* Health endpoints
* Readiness endpoints
* Easy containerization
* Integration with Kubernetes

---

## Why Streamlit?

Streamlit provides a simple interface for demonstrating the deployed ML system without building a separate frontend application.

It is particularly useful for:

* Model demonstrations
* Internal tools
* ML prototypes
* Stakeholder validation

---

## Why Docker?

Docker creates a consistent runtime environment containing:

* Application code
* Python runtime
* Dependencies
* ML framework
* Model artifacts
* Data required by the application

This reduces differences between development, CI, and production environments.

---

## Why Kubernetes?

Kubernetes provides:

* Container orchestration
* Replica management
* Service discovery
* Rolling deployments
* Health checks
* Horizontal scaling
* Self-healing
* Rollback capabilities

---

## Why GKE Autopilot?

GKE Autopilot reduces the operational burden of managing Kubernetes infrastructure.

Google manages much of the underlying node infrastructure while the application remains deployed using standard Kubernetes resources.

This allows the project to retain Kubernetes concepts without requiring manual node management.

---

## Why Artifact Registry?

Artifact Registry provides a centralized repository for Docker images.

Instead of deploying locally built images, the pipeline follows:

```text
GitHub Actions
      ↓
Docker Build
      ↓
Artifact Registry
      ↓
GKE
```

This creates a reproducible deployment path.

---

## Why Git SHA Image Tags?

Using:

```text
<git-sha>
```

instead of:

```text
latest
```

provides deployment traceability.

For example:

```text
Git commit ABC123
       ↓
Docker image ABC123
       ↓
GKE deployment ABC123
```

A previous image can therefore be identified and restored.

---

## Why Two API Replicas?

Two replicas provide:

* Basic availability
* Safer rolling updates
* Capacity for concurrent requests
* Reduced dependence on a single pod

The API deployment can subsequently scale up to five replicas using HPA.

---

## Why ClusterIP for the API?

The API is an internal service.

The dashboard is the public-facing component.

Therefore:

```text
Internet
   ↓
LoadBalancer
   ↓
Streamlit
   ↓
ClusterIP
   ↓
FastAPI
```

This reduces unnecessary public exposure of the inference API.

---

## Why Readiness Probes?

A model-serving application may require time to:

* Start Python
* Import TensorFlow
* Load the model
* Load the scaler
* Initialize the API

The readiness probe prevents Kubernetes from sending traffic to the pod until it is actually ready.

---

## Why Workload Identity Federation?

Long-lived service-account credentials introduce security risk.

Workload Identity Federation allows GitHub Actions to authenticate using short-lived credentials obtained through an OIDC trust relationship.

This eliminates the need to store a permanent GCP service-account key in the repository.

---

# Production Improvements

The current project demonstrates a complete production-style MLOps workflow.

For a larger enterprise deployment, the following improvements would be appropriate.

## Model Registry

Move:

```text
cnn_lstm_model.keras
```

from Git into a dedicated model registry or object storage system.

This would allow:

```text
Model v1
Model v2
Model v3
```

to be independently versioned.

---

## Separate Model and Application Lifecycles

Currently the model artifact is packaged with the application container.

A more mature architecture would separate:

```text
Application Version
        +
Model Version
```

For example:

```text
API v2
Model v17
```

rather than requiring an application rebuild for every model update.

---

## Data Pipeline

A production system could introduce:

```text
Sensor Data
     ↓
Streaming / Batch Ingestion
     ↓
Data Validation
     ↓
Feature Engineering
     ↓
Feature Store
     ↓
Model Inference
```

---

## Model Monitoring

A production implementation could monitor:

* Prediction distribution
* RUL distribution
* Input feature drift
* Model latency
* API error rate
* Prediction volume
* Data quality

---

## Centralized Logging

Application and Kubernetes logs could be centralized through cloud logging.

This would make production troubleshooting easier.

---

## Advanced Observability

The current project deliberately avoids adding Prometheus/Grafana.

A future enterprise implementation could introduce:

```text
Application
     ↓
Metrics
     ↓
Prometheus
     ↓
Grafana
```

or use managed cloud observability services.

---

## Automated Model Validation

The CI/CD pipeline could eventually include:

```text
Train Model
     ↓
Evaluate Model
     ↓
Compare Against Baseline
     ↓
Model Quality Gate
     ↓
Register Model
     ↓
Deploy
```

This prevents a model with degraded performance from reaching production.

---

# Project Outcome

The project successfully demonstrates a complete ML-to-production workflow:

```text
                     END-TO-END MLOPS

                         GitHub
                            │
                            ▼
                    GitHub Actions
                            │
                   ┌────────┴────────┐
                   │                 │
                pytest           Docker Build
                   │                 │
                   └────────┬────────┘
                            │
                            ▼
                    Artifact Registry
                            │
                            ▼
                     GKE Autopilot
                            │
               ┌────────────┴────────────┐
               │                         │
               ▼                         ▼
        Streamlit Dashboard         FastAPI × 2
          LoadBalancer                ClusterIP
               │                         │
               └──────────┬──────────────┘
                          │
                          ▼
                     CNN-LSTM
                          │
                          ▼
                   RUL Prediction
                          │
                          ▼
                    96.45 cycles
```

The system demonstrates:

* Machine learning
* Time-series prediction
* Model serving
* REST APIs
* Docker
* Kubernetes
* GKE Autopilot
* Artifact Registry
* GitHub Actions
* Workload Identity Federation
* Rolling deployments
* Readiness and liveness probes
* Horizontal scaling
* Deployment rollback
* Cloud-based ML inference

---

# License

This project is intended for educational, portfolio, and demonstration purposes.
