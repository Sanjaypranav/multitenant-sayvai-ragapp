# Multi-Tenant RAG

Multi-Tenant RAG (Retrieval-Augmented Generation) is a Python-based project designed to implement RAG workflows for multi-tenant architectures. It combines vector-based search, efficient data retrieval, and intelligent query handling tailored for multi-tenancy.

---

## Overview

Multi-Tenant RAG is a system designed to provide efficient and scalable solutions for retrieval-augmented generation in multi-tenant environments. It supports:

- **Multi-Tenant Architecture**: Isolates data and logic for each tenant.
- **Vector-Based Search**: Uses `Milvus` for storing and retrieving embeddings.
- **Document Splitting**: Automatically segments documents for better retrieval.
- **Extensible Design**: Easily integrates new features or external APIs.

---

## Installation

Follow these steps to set up and install Multi-Tenant RAG.

### Prerequisites

1. **Python**: Version 3.8 or above
2. **Poetry**: Dependency management tool
3. **Milvus**: Vector database (ensure it is installed and running)
4. [Docker](https://www.docker.com/get-started)
5. [Docker Compose](https://docs.docker.com/compose/install/)

---

## Steps to Install

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd multi-tenant-rag
# Set Up Environment Variables

Follow these steps to configure the environment for Multi-Tenant RAG:

---

## Steps

1. **Copy the Example Environment File**:
   ```bash
   cp .env.example .env

# Install Dependencies

To install the required dependencies for Multi-Tenant RAG, follow these steps:

---

## Steps

1. **Install Dependencies Using Poetry**:
   ```bash
   poetry install

# Usage

Follow these steps to run the Multi-Tenant RAG application.

---

## Run the Application

1. **Using the Batch Script** (Windows):
   ```bash
   ./run.bat

## Using the Dockerfile

The `Dockerfile` is used to create a Docker image for the **Multi-Tenant RAG** project.

### Steps to Build and Run with Dockerfile

1. **Build the Docker Image**
   Run the following command in the root directory of the project:
   ```bash
   docker build -t multi-tenant-rag .

### Start the Services
1. **Compose the Docker Image**
    Run the following command:
    ```bash
    docker-compose up --build


# Contributing

We welcome contributions to Multi-Tenant RAG! Follow the guidelines below to get started.

---

## Steps to Contribute

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch-name

