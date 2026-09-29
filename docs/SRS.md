# Software Requirements Specification (SRS)

## Project: Intelligent Intrusion Detection System (IIDS)

### Project Team

- **Aditya Kumar Laur** — B.Tech CSE (Artificial Intelligence & Machine Learning)
- **Aditya Ranjan Kumar** — B.Tech CSE (Cybersecurity)
- **Aakar Singh** — B.Tech CSE (Artificial Intelligence & Machine Learning)
- **Vikash Kumar** — B.Tech CSE (Cybersecurity)

---

## 1. Introduction

### 1.1 Purpose
The purpose of this project is to develop and implement an Intelligent Intrusion Detection System (IIDS) using a hybrid machine learning pipeline. The system provides real-time detection, forensic analysis, and automated mitigation capabilities against unauthorized network access and malicious traffic patterns.

### 1.2 Scope
- Automated feature scaling and preprocessing of NSL-KDD benchmark traffic.
- Hybrid decision engine coupling Random Forest with 1D-CNN + LSTM.
- Dynamic confidence threshold routing ($\theta = 0.85$).
- Persistent SQLite forensic alert storage and incident logging.
- Single-packet evaluation and batch CSV scanning interfaces.
- Responsive web-based operations console for security operators.

---

## 2. Overall Description

### 2.1 User Classes and Characteristics
- **Security Analyst / SOC Operator**: Monitors real-time telemetry, analyzes threat breakdown charts, inspects forensic alert logs, and enforces manual IP blocks.
- **System Administrator**: Manages system configuration, selects inference engines (Hybrid / Random Forest only / CNN-LSTM only), and configures detection confidence cutoffs.

### 2.2 Operating Environment
- **Operating System**: Cross-platform (Linux / macOS / Windows).
- **Runtime**: Python 3.10+ with Flask backend, SQLite 3 database.
- **Frontend Runtime**: Modern web browser (Chrome, Edge, Firefox, Safari) running ECMAScript 2020+.

---

## 3. System Features & Functional Requirements

### 3.1 Machine Learning Classification Engine
- **FR-1.1**: The system shall classify connection records into five classes: `Normal`, `DoS`, `Probe`, `R2L`, and `U2R`.
- **FR-1.2**: The Random Forest classifier shall serve as the primary inference stage.
- **FR-1.3**: When Random Forest classification confidence is below 85%, the system shall route the record to the CNN-LSTM deep learning model.
- **FR-1.4**: The system shall expose configurable engine overrides (`hybrid`, `rf_only`, `cnn_only`).

### 3.2 Real-Time Telemetry & Batch Processing
- **FR-2.1**: The system shall provide a simulated live network feed streaming records from evaluation data pools.
- **FR-2.2**: The system shall support bulk CSV file uploads up to 50,000 records per scan.
- **FR-2.3**: Uploaded scan results shall produce aggregate metrics (normal traffic vs threat breakdown).

### 3.3 Security & Incident Response
- **FR-3.1**: Every flagged intrusion event shall be logged with source/destination IPs, timestamp, classification, and confidence.
- **FR-3.2**: Operators shall have the ability to blacklist offending source IPs with an associated justification.
- **FR-3.3**: The inspection pipeline shall reject incoming packets originating from blacklisted IPs before executing model inference.

---

## 4. Non-Functional Requirements

### 4.1 Performance & Latency
- Primary Random Forest evaluation latency shall not exceed 5ms per single record.
- Hybrid fallback inference using CNN-LSTM shall conclude within 50ms per single record.

### 4.2 Usability & Reliability
- The operations dashboard shall maintain responsive rendering with dark/light theme switching.
- Database operations shall use transactional SQLite connections with proper connection cleanup.
