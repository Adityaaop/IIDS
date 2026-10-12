# Intelligent Intrusion Detection System (IIDS)

An intrusion detection system that uses a hybrid machine learning pipeline to classify network traffic into normal activity or attack categories (DoS, Probe, R2L, U2R).

The system uses a two-stage approach:
1. Fast classification using a Random Forest model for high-confidence predictions (< 5ms latency).
2. Deep sequential analysis using a 1D-CNN + LSTM network when confidence falls below the 0.85 threshold.

---

## Contributors

- **Aditya Ranjan Kumar** ([@Adityaaop](https://github.com/Adityaaop)) - Frontend dashboard, real-time Chart.js telemetry, operator session controls
- **Aditya Kumar Laur** ([@adityaklaur](https://github.com/adityaklaur)) - Flask backend API, hybrid model decision pipeline, Docker setup, pytest test suite
- **Aakar Singh** ([@aakarrr](https://github.com/aakarrr)) - NSL-KDD dataset preprocessing, feature scaling (StandardScaler), 1D-CNN + LSTM model
- **Vikash Kumar** ([@hey-sirii](https://github.com/hey-sirii)) - SQLite forensics database, attack taxonomy mapping, IP blocking mitigation

---

## Architecture

```
                      +-----------------------------+
                      |   Network Traffic Vector    |
                      |   (41 NSL-KDD Features)     |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   StandardScaler Normalize  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Stage 1: Random Forest      |
                      | Classifier                  |
                      +--------------+--------------+
                                     |
                             Confidence >= 0.85?
                               /            \
                         YES  /              \  NO (< 0.85)
                             v                v
              +----------------------+   +-----------------------+
              | Direct RF Prediction |   | Stage 2: CNN + LSTM   |
              | (< 5ms response)     |   | Deep Temporal Feature |
              |                      |   | Extraction & Fallback |
              +----------+-----------+   +-----------+-----------+
                         |                           |
                         +-------------+-------------+
                                       |
                                       v
                       +-------------------------------+
                       | Final Prediction & Confidence |
                       | (Normal, DoS, Probe, R2L, U2R)|
                       +---------------+---------------+
                                       |
                                       v
                       +-------------------------------+
                       | SQLite Forensic Log & SOC UI  |
                       +-------------------------------+
```

For more details on mathematical formulation and network layers, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Features

- **Hybrid Classification**: Random Forest handles clear cases quickly, while CNN-LSTM handles uncertain cases below 0.85 confidence.
- **5 Attack Categories**:
  - `Normal`: Legitimate connection.
  - `DoS`: Denial of service (Neptune, Smurf, Pod, Teardrop, etc.).
  - `Probe`: Port scans and network probing (Nmap, Portsweep, IPsweep, etc.).
  - `R2L`: Unauthorized remote access attempts (Guess_passwd, Warezclient, etc.).
  - `U2R`: Local privilege escalation attempts (Buffer_overflow, Rootkit, etc.).
- **Live SOC Dashboard**: Web interface with real-time Chart.js charts, alert logs, and dark theme.
- **Batch CSV Scanner**: Upload connection logs for batch classification and summary statistics.
- **Forensic Logging & Mitigation**: Events stored in SQLite (`iids_forensics.db`) with timestamps, source IPs, and predicted attack type. Malicious IPs can be blocked directly from the dashboard.

---

## Model Benchmark Performance

Trained and evaluated on the NSL-KDD benchmark dataset:

| Model | Test Accuracy | Precision | Recall | F1-Score | Latency |
|---|---|---|---|---|---|
| **Random Forest** | **99.66%** | 0.996 | 0.997 | 0.996 | < 5ms (Primary) |
| **CNN-LSTM** | **99.09%** | 0.991 | 0.991 | 0.991 | ~35ms (Fallback) |

---

## Project Structure

```
├── backend/
│   └── app.py                     # Flask API, authentication, and database handlers
├── data/
│   └── KDDTrain+_20Percent.txt    # NSL-KDD training and test dataset subset
├── docs/
│   ├── ARCHITECTURE.md            # System architecture and model formulation
│   ├── API_REFERENCE.md           # API endpoints reference
│   └── SRS.md                     # Software requirements specification
├── frontend/
│   ├── index.html                 # Dashboard UI (HTML, CSS, JavaScript, Chart.js)
│   └── vercel.json                # Routing and deployment configuration
├── model/
│   ├── cnn_lstm_model.h5          # Trained CNN-LSTM model weights
│   ├── rf_model.pkl               # Trained Random Forest classifier
│   ├── scaler.pkl                 # Fitted StandardScaler object
│   ├── train_pipeline.py          # Training and preprocessing script
│   ├── data_visualization.py      # Plotting scripts
│   ├── rf_confusion_matrix.png    # Evaluation confusion matrix (RF)
│   ├── cnn_lstm_confusion_matrix.png # Evaluation confusion matrix (CNN-LSTM)
│   └── roc_curve.png              # ROC evaluation curves
├── tests/
│   ├── test_api.py                # API endpoint unit tests
│   └── test_model.py              # Model artifact tests
├── Dockerfile                     # Docker container build file
├── docker-compose.yml             # Docker compose setup
├── requirements.txt               # Dependencies
├── CONTRIBUTING.md                # Contribution guidelines
├── SECURITY.md                    # Security reporting
└── LICENSE                        # MIT License
```

---

## Getting Started

### Prerequisites
- Python 3.10+ (or Docker)
- pip

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Adityaaop/IIDS.git
   cd IIDS
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Start the application:**
   ```bash
   python backend/app.py
   ```

4. **Access the dashboard:**
   Open [http://localhost:5001](http://localhost:5001) in your browser.
   - **Username**: `admin`
   - **Password**: `admin`

---

### Running with Docker

```bash
docker compose up --build -d
```
The app will be available at [http://localhost:5001](http://localhost:5001).

To stop the container:
```bash
docker compose down
```

---

## Running Tests

Run the test suite with `pytest`:

```bash
pytest tests/ -v
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | System health check and model loading state |
| `POST` | `/api/login` | Operator authentication |
| `POST` | `/api/predict` | Single packet prediction |
| `GET` | `/api/live_feed` | Simulated packet stream with real-time classification |
| `POST` | `/api/upload_scan` | Batch CSV upload and threat detection |
| `GET` | `/api/history` | Query forensic alert history |
| `GET` / `POST` | `/api/blocked` | View or add blocked IPs |
| `GET` / `POST` | `/api/settings` | Get or update engine mode (`hybrid`, `rf_only`, `cnn_only`) |

For full request and response schemas, see [docs/API_REFERENCE.md](docs/API_REFERENCE.md).

---

## License

This project is licensed under the [MIT License](LICENSE).
