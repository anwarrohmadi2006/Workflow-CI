# Workflow-CI

## 📋 Deskripsi
Repository untuk **K3 - Continuous Integration** pada submission Dicoding MLOps. Berisi CI/CD pipeline untuk build dan push Docker image ke DockerHub.

## 📁 Struktur Folder
```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI workflow
├── MLProject/
│   ├── Dockerfile            # Docker image definition
│   ├── modelling.py          # Model training script
│   ├── requirements.txt      # Dependencies
│   └── DockerHub.txt         # Link ke DockerHub
├── DockerHub.txt             # Link ke DockerHub image
└── README.md                 # Dokumentasi
```

## 🚀 CI/CD Pipeline

### GitHub Actions Workflow
```yaml
name: ML CI Pipeline
on: [push]
jobs:
  build:
    - Train model with MLflow
    - Build Docker image
    - Push to DockerHub
```

### Trigger
- Setiap push ke repository akan trigger workflow

## 🐳 Docker

### Build Manual
```bash
cd MLProject
docker build -t house-price-model .
```

### Pull dari DockerHub
```bash
docker pull anwarrohmadi/house-price-model:latest
```

## 📦 DockerHub Image
Link: [DockerHub Repository](https://hub.docker.com/r/anwarrohmadi/house-price-model)

## 👤 Author
**Anwar Rohmadi**

## 🔗 Links
- [GitHub Repository](https://github.com/anwarrohmadi2006/Workflow-CI)
- [DockerHub Image](https://hub.docker.com/r/anwarrohmadi/house-price-model)
