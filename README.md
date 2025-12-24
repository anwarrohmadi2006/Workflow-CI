# 🐳 MLOps Continuous Integration Pipeline (K3)
**Project Title**: Automated Model Training & Containerization Management  
**Author**: Anwar Rohmadi  
**Organization**: Dicoding Academy - Membangun Sistem Machine Learning

---

## 📋 Project Overview
This repository manages the Continuous Integration (CI) lifecycle for the House Price Prediction model. It automates the transition from preprocessed data to a production-ready Docker image, ensuring consistent model performance and deployment reliability.

## 📁 Repository Structure
The repository is structured to support MLflow's standard deployment conventions:

```text
Workflow-CI/
├── .github/workflows/       # GitHub Actions CI/CD Pipeline
│   └── ci.yml               # Orchestrates Training, Build, and Registry Push
├── MLProject/               # MLflow Standard Component Folder
│   ├── modelling.py         # Advanced model training script
│   ├── conda.yaml           # Environment & Dependency specifications
│   ├── MLProject            # MLflow entry-point configuration
│   ├── Dockerfile           # Derived container image definition
│   └── DockerHub.txt        # Verified Docker Hub Image Link
├── README.md                # Technical Documentation
└── .workflow/               # (Optional) Workflow metadata
```

## 🛠️ Phase 1: Automated Training (Skilled)
The system utilizes **MLflow** for robust lifecycle management:
- **Reproducibility**: Environment defined in `conda.yaml`.
- **Logic**: `modelling.py` integrates with **DagsHub** for remote experiment tracking.
- **Workflow**: Automated via the `mlops-pipeline.yml` (located in the root submission) which pulls preprocessed data from Phase 1.

## 📦 Phase 2: Containerization & Registry (Advance)
The pipeline automatically packages the trained model into a production-ready Docker container:
1. **Build**: Converts the MLflow model artifact into a standalone Docker image using `mlflow models build-docker`.
2. **Push**: Authenticates and pushes the image to **Docker Hub**.
3. **Traceability**: Each image is tagged with the specific GitHub commit SHA for auditability.

**Docker Hub Image**: [anwarrohmadi111784/mlops-dicoding-model](https://hub.docker.com/r/anwarrohmadi111784/mlops-dicoding-model)

## 🚀 DevOps Workflow
The CI pipeline is triggered by commits to the `master` branch:
- **Stage 1**: Environment setup and dependency installation.
- **Stage 2**: Model training with hyperparameter logging.
- **Stage 3**: Containerization.
- **Stage 4**: Deployment to the Docker Hub registry.

---
*This work demonstrates full automation of the model lifecycle, from code commit to cloud-ready container.*
