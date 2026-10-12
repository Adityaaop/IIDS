# IIDS REST API Reference

The backend exposes a REST API powered by Flask for real-time inference, log batch processing, threat history, and administrative controls.

Base URL: `http://localhost:5001`

---

## 1. System & Authentication

### `GET /api/status`
Returns service health, loaded ML models, and dataset availability.
- **Response**:
  ```json
  {
    "status": "online",
    "rf_loaded": true,
    "cnn_loaded": true,
    "scaler_loaded": true,
    "total_alerts": 142
  }
  ```

### `POST /api/login`
Authenticates an administrator session.
- **Payload**:
  ```json
  {
    "username": "admin",
    "password": "your_secure_password"
  }
  ```
- **Response**: `200 OK` with session cookie.

### `POST /api/logout`
Terminates active session and clears auth cookies.

---

## 2. Detection & Inference

### `POST /api/predict`
Evaluates a single network packet record (41 features).
- **Payload**:
  ```json
  {
    "features": [0, "tcp", "http", "SF", 181, 5450, 0, ...],
    "engine": "hybrid"
  }
  ```
- **Response**:
  ```json
  {
    "prediction": "Normal",
    "confidence": 0.984,
    "engine_used": "Random Forest",
    "timestamp": "2026-10-12T16:30:00Z"
  }
  ```

### `GET /api/live_feed`
Generates simulated real-time connection traffic from the preprocessed evaluation pool.
- **Query Parameters**:
  - `engine` (optional): `hybrid` | `rf_only` | `cnn_only` (default: `hybrid`)
- **Response**:
  ```json
  {
    "timestamp": "2026-10-12T16:30:05Z",
    "src_ip": "192.168.1.105",
    "dst_ip": "10.0.0.1",
    "protocol": "tcp",
    "service": "http",
    "prediction": "DoS",
    "confidence": 0.991,
    "engine_used": "Random Forest",
    "blocked": false
  }
  ```

### `POST /api/upload_scan`
Uploads a batch CSV file for automated bulk inspection.
- **Form Data**:
  - `file`: CSV file containing network connection records.
- **Response**:
  ```json
  {
    "total_records": 500,
    "normal_count": 390,
    "threats_detected": 110,
    "breakdown": {
      "DoS": 85,
      "Probe": 20,
      "R2L": 4,
      "U2R": 1
    }
  }
  ```

---

## 3. Threat Forensics & Active Mitigation

### `GET /api/history`
Retrieves past alert records from SQLite.
- **Query Parameters**:
  - `limit`: Maximum number of records to return (default: `100`).
- **Response**:
  ```json
  [
    {
      "id": 1,
      "timestamp": "2026-10-12T16:25:12Z",
      "src_ip": "172.16.0.42",
      "protocol": "tcp",
      "service": "private",
      "attack_type": "Neptune",
      "attack_category": "DoS",
      "confidence": 0.996,
      "engine": "Random Forest"
    }
  ]
  ```

### `GET /api/blocked`
Returns list of currently blocked source IP addresses.

### `POST /api/blocked`
Adds an IP to the active blocklist.
- **Payload**:
  ```json
  {
    "ip": "192.168.1.180",
    "reason": "Repeated DoS SYN flood detected"
  }
  ```

### `DELETE /api/blocked/<ip>`
Removes the specified IP from the firewall blocklist.

---

## 4. Configuration

### `GET /api/settings`
Returns current engine routing settings and threshold configuration.

### `POST /api/settings`
Updates operational parameters (engine mode, alert confidence threshold).
- **Payload**:
  ```json
  {
    "engine": "hybrid",
    "threshold": 0.85
  }
  ```
