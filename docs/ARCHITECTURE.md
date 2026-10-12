# System Architecture: Intelligent Intrusion Detection System (IIDS)

## 1. High-Level Overview

IIDS is a hybrid machine-learning intrusion detection system designed to inspect network traffic records and detect potential cyber attacks with low latency and high accuracy. 

The architecture combines a fast, tree-based classifier (**Random Forest**) with a deep sequence model (**CNN-LSTM**) via confidence-based threshold routing.

```
                      +-----------------------------+
                      |   Network Traffic Feature   |
                      |   Vector (41 NSL-KDD Atts)  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   StandardScaler Normalizer |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Stage 1: Random Forest     |
                      |      Classifier             |
                      +--------------+--------------+
                                     |
                     Confidence >= 0.85?
                       /            \
                 YES  /              \  NO (Uncertain)
                     v                v
      +----------------------+   +-----------------------+
      | Direct RF Prediction |   | Stage 2: 1D-CNN + LSTM|
      | (Sub-millisecond)    |   | Deep Temporal Feature |
      |                      |   | Extraction & Fallback |
      +----------+-----------+   +-----------+-----------+
                 |                           |
                 +-------------+-------------+
                               |
                               v
               +-------------------------------+
               |  Predicted Class & Confidence |
               | (Normal, DoS, Probe, R2L, U2R)|
               +---------------+---------------+
                               |
                               v
               +-------------------------------+
               | SQLite Forensics Persistence  |
               | & Web Dashboard Telemetry     |
               +-------------------------------+
```

---

## 2. Model Pipeline

### 2.1 Feature Preprocessing
- **Dataset**: NSL-KDD Benchmark (derived from KDDCup99, eliminating redundant and duplicate connection records).
- **Features (41 attributes)**:
  - *Basic features*: `duration`, `protocol_type`, `service`, `flag`, `src_bytes`, `dst_bytes`, `land`, `wrong_fragment`, `urgent`.
  - *Content features*: `hot`, `num_failed_logins`, `logged_in`, `num_compromised`, `root_shell`, `su_attempted`, `num_root`, `num_file_creations`, `num_shells`, `num_access_files`, `num_outbound_cmds`, `is_host_login`, `is_guest_login`.
  - *Time-based traffic features*: `count`, `srv_count`, `serror_rate`, `srv_serror_rate`, `rerror_rate`, `srv_rerror_rate`, `same_srv_rate`, `diff_srv_rate`, `srv_diff_host_rate`.
  - *Host-based traffic features*: `dst_host_count`, `dst_host_srv_count`, `dst_host_same_srv_rate`, `dst_host_diff_srv_rate`, `dst_host_same_src_port_rate`, `dst_host_srv_diff_host_rate`, `dst_host_serror_rate`, `dst_host_srv_serror_rate`, `dst_host_rerror_rate`, `dst_host_srv_rerror_rate`.
- **Encoding & Normalization**:
  - Categorical variables (`protocol_type`, `service`, `flag`) are encoded into integer indices.
  - Continuous metrics are transformed using `StandardScaler` fitted on the training split to produce zero-mean, unit-variance distributions.

### 2.2 Attack Category Mapping
Fine-grained attack labels in NSL-KDD are mapped into 5 primary taxonomies:
1. **Normal**: Legitimate, non-malicious network activity.
2. **DoS (Denial of Service)**: Flooding target servers to exhaust system resources (e.g., Neptune, Smurf, Pod, Teardrop).
3. **Probe**: Port scanning and network reconnaissance (e.g., Nmap, Portsweep, IPsweep, Satan).
4. **R2L (Remote to Local)**: Unauthorized access gained remotely through password guessing or network exploits (e.g., Guess_passwd, Warezclient).
5. **U2R (User to Root)**: Privilege escalation from local unprivileged account to superuser (e.g., Buffer_overflow, Rootkit).

### 2.3 Hybrid Decision Logic
1. Input feature vector is preprocessed and passed to the Random Forest model.
2. The Random Forest generates predicted class probabilities across all 5 categories.
3. If the maximum predicted probability $\ge 0.85$:
   - The Random Forest prediction is returned immediately.
4. If the confidence $< 0.85$:
   - The vector is reshaped into sequence format `(batch_size, 41, 1)` and forwarded to the CNN-LSTM network.
   - 1D Convolution filters extract localized spatial correlations, followed by LSTM recurrent cells capturing temporal dependencies across the feature series.
   - The CNN-LSTM model's softmax distribution provides the final classification.

---

## 3. Storage & Incident Response

### 3.1 SQLite Forensics Storage (`iids_forensics.db`)
- Every detection event is recorded with:
  - Timestamp (UTC ISO-8601)
  - Source IP / Destination IP
  - Protocol & Service
  - Attack Category & Specific Label
  - Severity level & Decision Engine (RF vs CNN-LSTM)
  - Confidence Score

### 3.2 Mitigation Controls
- Operators can actively block hostile source IPs via `/api/blocked`.
- The live inspection pipeline verifies incoming traffic against active firewall rules stored in `blocked_ips` before running inference.
