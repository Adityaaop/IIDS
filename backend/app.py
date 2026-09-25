"""Flask backend for the Intelligent Intrusion Detection System (IIDS).

This backend loads the trained ML artifacts, prepares a small live traffic
simulation pool from the NSL-KDD dataset, and exposes a hybrid inference route
that uses the Random Forest model first and falls back to CNN-LSTM when the RF
confidence is not strong enough.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import secrets
from datetime import datetime, timezone
from functools import wraps
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, session
from flask_cors import CORS 
try:
	from tensorflow.keras.models import load_model  # type: ignore
except ImportError:
	load_model = None
from werkzeug.security import check_password_hash, generate_password_hash

# Initialize Flask App and Enable CORS
app = Flask(__name__)
CORS(app, supports_credentials=True)

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, os.pardir))
MODEL_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "model"))
FRONTEND_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "frontend"))
DATA_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, "data", "KDDTrain+_20Percent.txt"))
DB_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, "iids_forensics.db"))
SECRET_KEY_FILE = os.path.abspath(os.path.join(BACKEND_DIR, ".iids_secret_key"))
USERS_TABLE = "users"
SETTINGS_TABLE = "settings"
LEGACY_SETTINGS_TABLE = "app_settings"
ALERTS_TABLE = "alerts"
BLOCKED_IPS_TABLE = "blocked_ips"
DEFAULT_ENGINE_MODE = "hybrid"
ENGINE_MODES = {"hybrid", "rf_only", "cnn_only"}
ENGINE_ALIASES = {
	"hybrid": "hybrid",
	"h": "hybrid",
	"rf": "rf_only",
	"rf_only": "rf_only",
	"random_forest": "rf_only",
	"random forest": "rf_only",
	"cnn": "cnn_only",
	"cnn_only": "cnn_only",
	"cnn-lstm": "cnn_only",
	"cnn_lstm": "cnn_only",
}
DEFAULT_THEME = "dark"
DEFAULT_ALERT_THRESHOLD = 0.85
DEFAULT_ADMIN_USERNAME = os.environ.get("IIDS_ADMIN_USERNAME", "admin")
DEFAULT_ADMIN_PASSWORD = os.environ.get("IIDS_ADMIN_PASSWORD", "admin")
DEFAULT_ADMIN_PASSWORD_HASH = os.environ.get(
	"IIDS_ADMIN_PASSWORD_HASH",
	generate_password_hash(DEFAULT_ADMIN_PASSWORD),
)
CONFIDENCE_THRESHOLD = DEFAULT_ALERT_THRESHOLD
MAX_UPLOAD_ROWS = 50000

if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)


def get_persistent_secret_key() -> str:
	"""Load a stable Flask secret key from the environment or disk."""

	env_key = os.environ.get("IIDS_SECRET_KEY")
	if env_key:
		return env_key

	try:
		if os.path.exists(SECRET_KEY_FILE):
			with open(SECRET_KEY_FILE, "r", encoding="utf-8") as secret_file:
				stored_key = secret_file.read().strip()
				if stored_key:
					return stored_key

		secret_key = secrets.token_urlsafe(64)
		os.makedirs(os.path.dirname(SECRET_KEY_FILE), exist_ok=True)
		with open(SECRET_KEY_FILE, "w", encoding="utf-8") as secret_file:
			secret_file.write(secret_key)
		try:
			os.chmod(SECRET_KEY_FILE, 0o600)
		except (AttributeError, OSError, PermissionError):
			pass
		return secret_key
	except OSError:
		return "iids-production-secret-key-stable-2026"


# -----------------------------------------------------------------------------
# Flask application
# -----------------------------------------------------------------------------
# The frontend folder serves double duty here: it is used for templates and for
# static assets so index.html can be rendered directly from the project UI.
app = Flask(__name__, template_folder=FRONTEND_DIR, static_folder=FRONTEND_DIR)
app.config["SECRET_KEY"] = get_persistent_secret_key()
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("SESSION_COOKIE_SECURE", "false").lower() in ("true", "1", "yes")


# -----------------------------------------------------------------------------
# NSL-KDD schema and label mapping
# -----------------------------------------------------------------------------
NSL_KDD_COLUMNS = [
	"duration",
	"protocol_type",
	"service",
	"flag",
	"src_bytes",
	"dst_bytes",
	"land",
	"wrong_fragment",
	"urgent",
	"hot",
	"num_failed_logins",
	"logged_in",
	"num_compromised",
	"root_shell",
	"su_attempted",
	"num_root",
	"num_file_creations",
	"num_shells",
	"num_access_files",
	"num_outbound_cmds",
	"is_host_login",
	"is_guest_login",
	"count",
	"srv_count",
	"serror_rate",
	"srv_serror_rate",
	"rerror_rate",
	"srv_rerror_rate",
	"same_srv_rate",
	"diff_srv_rate",
	"srv_diff_host_rate",
	"dst_host_count",
	"dst_host_srv_count",
	"dst_host_same_srv_rate",
	"dst_host_diff_srv_rate",
	"dst_host_same_src_port_rate",
	"dst_host_srv_diff_host_rate",
	"dst_host_serror_rate",
	"dst_host_srv_serror_rate",
	"dst_host_rerror_rate",
	"dst_host_srv_rerror_rate",
	"label",
	"difficulty",
]


# Map the fine-grained NSL-KDD attack names into the 5 high-level classes used
# by the project. This is the same semantic mapping used during training.
ATTACK_MAP = {
	"normal": "Normal",
	"neptune": "DoS",
	"back": "DoS",
	"land": "DoS",
	"pod": "DoS",
	"smurf": "DoS",
	"teardrop": "DoS",
	"mailbomb": "DoS",
	"apache2": "DoS",
	"processtable": "DoS",
	"udpstorm": "DoS",
	"worm": "DoS",
	"ipsweep": "Probe",
	"nmap": "Probe",
	"portsweep": "Probe",
	"satan": "Probe",
	"mscan": "Probe",
	"saint": "Probe",
	"guess_passwd": "R2L",
	"ftp_write": "R2L",
	"imap": "R2L",
	"phf": "R2L",
	"multihop": "R2L",
	"warezmaster": "R2L",
	"warezclient": "R2L",
	"spy": "R2L",
	"xlock": "R2L",
	"xsnoop": "R2L",
	"snmpguess": "R2L",
	"snmpgetattack": "R2L",
	"httptunnel": "R2L",
	"sendmail": "R2L",
	"named": "R2L",
	"buffer_overflow": "U2R",
	"loadmodule": "U2R",
	"rootkit": "U2R",
	"perl": "U2R",
	"sqlattack": "U2R",
	"xterm": "U2R",
	"ps": "U2R",
}


# LabelEncoder sorts labels alphabetically, so this is the encoded index order
# produced by the training pipeline.
CLASS_NAMES = ["DoS", "Normal", "Probe", "R2L", "U2R"]


# -----------------------------------------------------------------------------
# Global runtime artifacts
# -----------------------------------------------------------------------------
scaler = None
rf_model = None
cnn_lstm_model = None
feature_columns = None
sample_scaled_features = None
sample_protocol_types = None
models_loaded = False
sample_data_loaded = False
active_engine = DEFAULT_ENGINE_MODE


def load_sample_data() -> tuple[np.ndarray, np.ndarray]:
	"""Load and preprocess the first 1000 NSL-KDD rows for live simulation.

	The preprocessing mirrors the training pipeline:
	- drop the difficulty column
	- map attack labels to the 5 major classes
	- one-hot encode protocol/service/flag
	- align the resulting feature columns to the saved scaler
	- scale the features using the fitted StandardScaler

	The function returns the scaled feature matrix and the raw protocol_type values
	for each row so the live feed can expose the original traffic protocol.
	"""

	if scaler is None:
		raise RuntimeError("Scaler must be loaded before sample data can be prepared.")

	# Load a small slice of data for a lightweight live traffic pool.
	sample_frame = pd.read_csv(
		DATA_PATH,
		header=None,
		names=NSL_KDD_COLUMNS,
		nrows=1000,
	)

	# Remove the difficulty score immediately, matching the training pipeline.
	sample_frame = sample_frame.drop(columns=["difficulty"])

	# Keep the raw protocol values for the API response before we one-hot encode.
	raw_protocol_types = sample_frame["protocol_type"].astype(str).to_numpy()

	# Apply the exact same NSL-KDD attack grouping used during training.
	raw_labels = sample_frame["label"].astype(str)
	sample_frame["label"] = (
		raw_labels.str.strip().str.lower().map(ATTACK_MAP)
	)

	if sample_frame["label"].isna().any():
		unknown_labels = sorted(raw_labels[sample_frame["label"].isna()].unique())
		raise ValueError(f"Unmapped NSL-KDD labels found in sample pool: {unknown_labels}")

	# Separate the inputs from the target label. The live feed only needs features.
	features = sample_frame.drop(columns=["label"])

	# Perform one-hot encoding on the categorical network fields.
	features = pd.get_dummies(
		features,
		columns=["protocol_type", "service", "flag"],
		drop_first=False,
	)

	# Align the sample feature matrix to the exact column layout the scaler saw
	# during training. Missing columns are filled with zero; extra columns are
	# discarded. This keeps inference shape-safe.
	if feature_columns is not None:
		features = features.reindex(columns=feature_columns, fill_value=0)
	else:
		feature_columns_local = list(getattr(scaler, "feature_names_in_", features.columns))
		features = features.reindex(columns=feature_columns_local, fill_value=0)

	# Apply the fitted scaler to the live sample pool.
	scaled_features = scaler.transform(features)
	return scaled_features, raw_protocol_types


def generate_fake_ip() -> str:
	"""Create a simple fake private network address for the simulation feed."""

	return f"192.168.{np.random.randint(0, 256)}.{np.random.randint(1, 255)}"


def decode_class_index(class_index: int) -> str:
	"""Map a model output index back to the human-readable attack class."""

	if class_index < 0 or class_index >= len(CLASS_NAMES):
		raise ValueError(f"Invalid class index returned by model: {class_index}")
	return CLASS_NAMES[class_index]


def get_numeric_setting(setting_key: str, default: float) -> float:
	"""Read a numeric setting from SQLite and normalize it to float."""

	try:
		return float(get_setting(setting_key, str(default)))
	except (TypeError, ValueError):
		return float(default)


def get_alert_threshold() -> float:
	"""Return the persisted alert threshold used by hybrid routing."""

	threshold = get_numeric_setting("alert_threshold", DEFAULT_ALERT_THRESHOLD)
	return max(0.0, min(1.0, threshold))


def is_authenticated() -> bool:
	"""Return whether the current session is authenticated."""

	return bool(session.get("authenticated"))


def require_authentication(view_func):
	"""Protect API routes with a JSON 401 response."""

	@wraps(view_func)
	def wrapper(*args, **kwargs):
		if not is_authenticated():
			return jsonify({"status": "error", "message": "Authentication required."}), 401
		return view_func(*args, **kwargs)

	return wrapper


def get_user_record(username: str) -> sqlite3.Row | None:
	"""Fetch a persisted user record, if present."""

	with get_db_connection() as connection:
		ensure_core_tables(connection)
		cursor = connection.execute(
			f"SELECT username, password_hash, role, is_active FROM {USERS_TABLE} WHERE username = ?",
			(username,),
		)
		return cursor.fetchone()


def verify_admin_credentials(username: str, password: str) -> bool:
	"""Verify a login against the persisted admin credentials."""

	user_row = get_user_record(username)
	if user_row is None:
		stored_username = get_setting("admin_username", DEFAULT_ADMIN_USERNAME)
		stored_password_hash = get_setting("admin_password_hash", DEFAULT_ADMIN_PASSWORD_HASH)
		if username != stored_username:
			return False
	else:
		if not int(user_row["is_active"]):
			return False
		stored_password_hash = str(user_row["password_hash"])

	try:
		return check_password_hash(stored_password_hash, password)
	except (ValueError, TypeError):
		# Handle legacy/corrupt values gracefully and avoid 500s on login.
		if isinstance(stored_password_hash, str) and stored_password_hash == password:
			return True
		return False


def update_runtime_settings(payload: dict[str, Any]) -> dict[str, Any]:
	"""Validate and persist dashboard settings coming from the UI."""

	updates: dict[str, Any] = {}

	if "engine" in payload:
		updates["engine"] = normalize_engine(str(payload["engine"]))
		set_active_engine(updates["engine"])
		set_setting("engine", updates["engine"])

	if "theme" in payload:
		theme = str(payload["theme"]).strip().lower()
		if theme not in {"light", "dark"}:
			raise ValueError("Theme must be 'light' or 'dark'.")
		updates["theme"] = theme
		set_setting("theme", theme)

	if "alert_threshold" in payload:
		try:
			threshold = float(payload["alert_threshold"])
		except (TypeError, ValueError) as exc:
			raise ValueError("Alert threshold must be a number between 0 and 1.") from exc
		if not 0.0 <= threshold <= 1.0:
			raise ValueError("Alert threshold must be between 0 and 1.")
		updates["alert_threshold"] = threshold
		set_setting("alert_threshold", f"{threshold:.4f}")

	return updates


def normalize_engine(engine: str | None, fallback: str = DEFAULT_ENGINE_MODE) -> str:
	"""Validate and normalize a routing engine selection."""

	engine_value = (engine or fallback).strip().lower().replace(" ", "_")
	engine_value = ENGINE_ALIASES.get(engine_value, engine_value)
	if engine_value not in ENGINE_MODES:
		raise ValueError(f"Unsupported engine '{engine}'. Expected one of: {sorted(ENGINE_MODES)}")
	return engine_value


def get_request_engine(default: str | None = None) -> str:
	"""Extract the engine from the current request in a consistent way."""

	json_payload = request.get_json(silent=True) or {}
	engine = request.args.get("engine") or request.form.get("engine") or json_payload.get("engine")
	return normalize_engine(engine, fallback=default or active_engine)


def get_db_connection() -> sqlite3.Connection:
	"""Create a SQLite connection configured for the app's forensics workload."""

	connection = sqlite3.connect(DB_PATH, timeout=10)
	connection.row_factory = sqlite3.Row
	connection.execute("PRAGMA foreign_keys = ON")
	connection.execute("PRAGMA busy_timeout = 5000")
	try:
		connection.execute("PRAGMA journal_mode = WAL")
		connection.execute("PRAGMA synchronous = NORMAL")
	except sqlite3.Error:
		pass
	return connection


def ensure_settings_table(connection: sqlite3.Connection) -> None:
	"""Create the settings table used to persist runtime configuration."""

	ensure_core_tables(connection)


def ensure_core_tables(connection: sqlite3.Connection) -> None:
	"""Create every database table the app depends on."""

	connection.execute(
		"""
		CREATE TABLE IF NOT EXISTS users (
			username TEXT PRIMARY KEY,
			password_hash TEXT NOT NULL,
			role TEXT NOT NULL DEFAULT 'admin',
			is_active INTEGER NOT NULL DEFAULT 1,
			created_at TEXT NOT NULL,
			updated_at TEXT NOT NULL
		)
		"""
	)
	connection.execute(
		"""
		CREATE TABLE IF NOT EXISTS settings (
			setting_key TEXT PRIMARY KEY,
			setting_value TEXT NOT NULL,
			updated_at TEXT NOT NULL
		)
		"""
	)
	connection.execute(
		"""
		CREATE TABLE IF NOT EXISTS app_settings (
			setting_key TEXT PRIMARY KEY,
			setting_value TEXT NOT NULL,
			updated_at TEXT NOT NULL
		)
		"""
	)
	connection.execute(
		"""
		CREATE TABLE IF NOT EXISTS alerts (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			timestamp TEXT NOT NULL,
			source_ip TEXT NOT NULL,
			protocol TEXT NOT NULL,
			classification TEXT NOT NULL,
			confidence REAL NOT NULL,
			model_used TEXT NOT NULL
		)
		"""
	)
	connection.execute(
		"""
		CREATE TABLE IF NOT EXISTS blocked_ips (
			ip TEXT PRIMARY KEY,
			reason TEXT,
			created_at TEXT NOT NULL
		)
		"""
	)


def _upsert_setting(connection: sqlite3.Connection, setting_key: str, value: str) -> None:
	"""Persist a setting in both the new and legacy settings tables."""

	updated_at = datetime.now(timezone.utc).isoformat()
	for table_name in (SETTINGS_TABLE, LEGACY_SETTINGS_TABLE):
		connection.execute(
			f"""
			INSERT INTO {table_name} (setting_key, setting_value, updated_at)
			VALUES (?, ?, ?)
			ON CONFLICT(setting_key) DO UPDATE SET
				setting_value = excluded.setting_value,
				updated_at = excluded.updated_at
			""",
			(setting_key, value, updated_at),
		)


def seed_default_runtime_data(connection: sqlite3.Connection) -> None:
	"""Seed the admin user and default dashboard settings."""

	updated_at = datetime.now(timezone.utc).isoformat()
	connection.execute(
		"""
		INSERT INTO users (username, password_hash, role, is_active, created_at, updated_at)
		VALUES (?, ?, ?, 1, ?, ?)
		ON CONFLICT(username) DO UPDATE SET password_hash = excluded.password_hash, updated_at = excluded.updated_at
		""",
		(
			DEFAULT_ADMIN_USERNAME,
			DEFAULT_ADMIN_PASSWORD_HASH,
			"admin",
			updated_at,
			updated_at,
		),
	)
	_upsert_setting(connection, "admin_username", DEFAULT_ADMIN_USERNAME)
	_upsert_setting(connection, "admin_password_hash", DEFAULT_ADMIN_PASSWORD_HASH)
	_upsert_setting(connection, "active_engine", DEFAULT_ENGINE_MODE)
	_upsert_setting(connection, "engine", DEFAULT_ENGINE_MODE)
	_upsert_setting(connection, "theme", DEFAULT_THEME)
	_upsert_setting(connection, "alert_threshold", f"{DEFAULT_ALERT_THRESHOLD:.4f}")


def get_setting(setting_key: str, default: str = DEFAULT_ENGINE_MODE) -> str:
	"""Read a persisted setting from SQLite."""

	try:
		with get_db_connection() as connection:
			ensure_core_tables(connection)
			cursor = connection.execute(
				f"SELECT setting_value FROM {SETTINGS_TABLE} WHERE setting_key = ?",
				(setting_key,),
			)
			row = cursor.fetchone()
			if row is None:
				legacy_cursor = connection.execute(
					f"SELECT setting_value FROM {LEGACY_SETTINGS_TABLE} WHERE setting_key = ?",
					(setting_key,),
				)
				legacy_row = legacy_cursor.fetchone()
				if legacy_row is None:
					return default
				_upsert_setting(connection, setting_key, str(legacy_row["setting_value"]))
				return str(legacy_row["setting_value"])
			return str(row["setting_value"])
	except sqlite3.Error:
		return default


def set_setting(setting_key: str, value: str) -> None:
	"""Persist a runtime setting to SQLite."""

	with get_db_connection() as connection:
		ensure_core_tables(connection)
		_upsert_setting(connection, setting_key, value)


def set_active_engine(engine: str) -> str:
	"""Update the in-memory and persisted engine configuration."""

	global active_engine
	active_engine = normalize_engine(engine)
	set_setting("active_engine", active_engine)
	set_setting("engine", active_engine)
	return active_engine


def get_active_engine() -> str:
	"""Return the current engine preference, falling back to persistence."""

	global active_engine
	if active_engine not in ENGINE_MODES:
		active_engine = DEFAULT_ENGINE_MODE
	persisted_engine = get_setting("active_engine", "")
	if not persisted_engine:
		persisted_engine = get_setting("engine", active_engine)
	active_engine = normalize_engine(persisted_engine, fallback=active_engine)
	return active_engine


def prepare_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
	"""Validate and align a raw NSL-KDD feature frame for inference."""

	if frame.empty:
		raise ValueError("Uploaded data frame is empty.")

	required_columns = [column for column in NSL_KDD_COLUMNS if column not in {"label", "difficulty"}]
	missing_columns = [column for column in required_columns if column not in frame.columns]
	if missing_columns:
		raise ValueError(f"Missing required NSL-KDD columns: {missing_columns}")

	unexpected_columns = sorted(
		set(frame.columns) - set(required_columns) - {"label", "difficulty"}
	)
	if unexpected_columns:
		raise ValueError(f"Unexpected columns present in upload: {unexpected_columns}")

	clean_frame = frame.copy()
	if "difficulty" in clean_frame.columns:
		clean_frame = clean_frame.drop(columns=["difficulty"])
	if "label" in clean_frame.columns:
		clean_frame = clean_frame.drop(columns=["label"])

	for column in ["protocol_type", "service", "flag"]:
		clean_frame[column] = clean_frame[column].astype(str).str.strip().str.lower()

	for column in [column for column in clean_frame.columns if column not in {"protocol_type", "service", "flag"}]:
		clean_frame[column] = pd.to_numeric(clean_frame[column], errors="raise")

	encoded_frame = pd.get_dummies(clean_frame, columns=["protocol_type", "service", "flag"], drop_first=False)
	if feature_columns is not None:
		encoded_frame = encoded_frame.reindex(columns=feature_columns, fill_value=0)
	else:
		encoded_frame = encoded_frame.reindex(
			columns=list(getattr(scaler, "feature_names_in_", encoded_frame.columns)), fill_value=0
		)

	return encoded_frame


def scale_uploaded_frame(frame: pd.DataFrame) -> np.ndarray:
	"""Prepare uploaded data for model inference."""

	if scaler is None:
		raise RuntimeError("Scaler is unavailable.")
	prepared_frame = prepare_feature_frame(frame)
	return scaler.transform(prepared_frame)


def reshape_for_cnn(row: np.ndarray) -> np.ndarray:
	"""Reshape a 2D row set into the CNN-LSTM input tensor."""

	if row.ndim == 1:
		row = row.reshape(1, -1)
	if row.ndim != 2:
		raise ValueError(f"Expected a 1D or 2D array, received shape {row.shape}.")
	return row.reshape(row.shape[0], 1, row.shape[1])


def predict_row(row: np.ndarray, engine: str) -> tuple[int, str, float, float | None]:
	"""Predict a single NSL-KDD sample using the selected engine."""

	if rf_model is None and cnn_lstm_model is None:
		raise RuntimeError("Model artifacts are unavailable.")

	engine_mode = normalize_engine(engine)
	if engine_mode == "cnn_only":
		if cnn_lstm_model is not None:
			cnn_probabilities = cnn_lstm_model.predict(reshape_for_cnn(row), verbose=0)[0]
			cnn_index = int(np.argmax(cnn_probabilities))
			cnn_confidence = float(np.max(cnn_probabilities))
			return cnn_index, "CNN-LSTM", cnn_confidence, None
		engine_mode = "rf_only"

	rf_probabilities = rf_model.predict_proba(row.reshape(1, -1))[0]
	rf_index = int(np.argmax(rf_probabilities))
	rf_confidence = float(np.max(rf_probabilities))

	if engine_mode == "rf_only" or rf_confidence >= get_alert_threshold() or cnn_lstm_model is None:
		return rf_index, "RF", rf_confidence, rf_confidence

	cnn_probabilities = cnn_lstm_model.predict(reshape_for_cnn(row), verbose=0)[0]
	cnn_index = int(np.argmax(cnn_probabilities))
	cnn_confidence = float(np.max(cnn_probabilities))
	return cnn_index, "CNN-LSTM", cnn_confidence, rf_confidence


def predict_batch(scaled_data: np.ndarray, engine: str) -> list[dict[str, Any]]:
	"""Predict a batch of samples using the requested engine mode."""

	results: list[dict[str, Any]] = []
	engine_mode = normalize_engine(engine)

	if engine_mode == "rf_only":
		probabilities = rf_model.predict_proba(scaled_data)
		for row_probabilities in probabilities:
			predicted_index = int(np.argmax(row_probabilities))
			confidence = float(np.max(row_probabilities))
			results.append(
				{
					"predicted_index": predicted_index,
					"model_used": "RF",
					"confidence": confidence,
					"rf_confidence": confidence,
				}
			)
		return results

	if engine_mode == "cnn_only":
		if cnn_lstm_model is not None:
			probabilities = cnn_lstm_model.predict(reshape_for_cnn(scaled_data), verbose=0)
			for row_probabilities in probabilities:
				predicted_index = int(np.argmax(row_probabilities))
				confidence = float(np.max(row_probabilities))
				results.append(
					{
						"predicted_index": predicted_index,
						"model_used": "CNN-LSTM",
						"confidence": confidence,
						"rf_confidence": None,
					}
				)
			return results
		engine_mode = "rf_only"

	for row in scaled_data:
		predicted_index, model_used, confidence, rf_confidence = predict_row(row, engine_mode)
		results.append(
			{
				"predicted_index": predicted_index,
				"model_used": model_used,
				"confidence": confidence,
				"rf_confidence": rf_confidence,
			}
		)

	return results


# -----------------------------------------------------------------------------
# Database initialization and forensic logging
# -----------------------------------------------------------------------------
def init_db():
	"""Initialize the SQLite forensics database on startup.

	Creates the alerts table if it doesn't exist. This table stores persistent
	logs of all predictions made by the system, including timestamps, source IPs,
	protocols, classifications, confidence scores, and which model made the call.
	"""

	try:
		with get_db_connection() as conn:
			ensure_core_tables(conn)
			seed_default_runtime_data(conn)
			conn.commit()
		print(f"Database initialized at {DB_PATH}")
	except Exception as exc:
		print(f"Error initializing database: {exc}")


def log_alert(timestamp: str, source_ip: str, protocol: str, classification: str, confidence: float, model_used: str):
	"""Log a single prediction result to the forensics database."""

	try:
		with get_db_connection() as conn:
			ensure_core_tables(conn)
			conn.execute(
				"""
				INSERT INTO alerts (timestamp, source_ip, protocol, classification, confidence, model_used)
				VALUES (?, ?, ?, ?, ?, ?)
				""",
				(timestamp, source_ip, protocol, classification, confidence, model_used),
			)
	except Exception as exc:
		print(f"Error logging alert to database: {exc}")


def get_blocked_ips() -> list[dict[str, str]]:
	"""Return all blocked IPs from the DB."""
	try:
		with get_db_connection() as conn:
			ensure_core_tables(conn)
			cursor = conn.execute("SELECT ip, reason, created_at FROM blocked_ips ORDER BY created_at DESC")
			return [dict(row) for row in cursor.fetchall()]
	except Exception:
		return []


def add_blocked_ip(ip: str, reason: str | None = None) -> None:
	"""Insert or update a blocked IP record."""
	with get_db_connection() as conn:
		ensure_core_tables(conn)
		conn.execute(
			"""
			INSERT INTO blocked_ips (ip, reason, created_at)
			VALUES (?, ?, ?)
			ON CONFLICT(ip) DO UPDATE SET reason = excluded.reason, created_at = excluded.created_at
			""",
			(ip, reason or "manual", datetime.now(timezone.utc).isoformat()),
		)


def remove_blocked_ip(ip: str) -> None:
	with get_db_connection() as conn:
		ensure_core_tables(conn)
		conn.execute("DELETE FROM blocked_ips WHERE ip = ?", (ip,))


def is_ip_blocked(ip: str) -> bool:
	try:
		with get_db_connection() as conn:
			ensure_core_tables(conn)
			cursor = conn.execute("SELECT 1 FROM blocked_ips WHERE ip = ?", (ip,))
			return cursor.fetchone() is not None
	except Exception:
		return False


# -----------------------------------------------------------------------------
# Model startup
# -----------------------------------------------------------------------------
# Load the trained scaler and models once at startup so each request can reuse
# the same in-memory artifacts.
try:
	scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
	rf_model_path = os.path.join(MODEL_DIR, "rf_model.pkl")
	cnn_lstm_model_path = os.path.join(MODEL_DIR, "cnn_lstm_model.h5")

	scaler = joblib.load(scaler_path)
	rf_model = joblib.load(rf_model_path)
	if load_model is not None and os.path.exists(cnn_lstm_model_path):
		try:
			cnn_lstm_model = load_model(cnn_lstm_model_path)
		except Exception as cnn_err:
			print(f"Notice: CNN-LSTM model could not be loaded ({cnn_err}), running RF primary engine.")
	feature_columns = list(getattr(scaler, "feature_names_in_", [])) or None
	models_loaded = True
	print("Model artifacts loaded successfully.")
except Exception as exc:
	print(f"Error loading backend artifacts: {exc}")

if models_loaded:
	try:
		# Build the live traffic pool after the scaler is available.
		sample_scaled_features, sample_protocol_types = load_sample_data()
		sample_data_loaded = True
		print("Live sample pool loaded successfully.")
	except Exception as exc:
		print(f"Error loading live sample pool: {exc}")

active_engine = get_active_engine()

# Initialize the forensics database on startup
try:
	init_db()
except Exception as exc:
	print(f"Error initializing forensics database: {exc}")


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/")
def index() -> str:
	"""Render the frontend entry page."""

	return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def api_status():
	"""Report backend health and model availability."""

	return jsonify(
		{
			"status": "online",
			"models_loaded": models_loaded,
			"sample_data_loaded": sample_data_loaded,
			"active_engine": get_active_engine(),
			"theme": get_setting("theme", DEFAULT_THEME),
			"alert_threshold": get_alert_threshold(),
			"message": "IIDS backend is running.",
		}
	)


@app.route("/api/blocked", methods=["GET", "POST"])
@require_authentication
def api_blocked():
	if request.method == "GET":
		return jsonify({"status": "success", "blocked": get_blocked_ips()})

	payload = request.get_json(silent=True) or {}
	ip = str(payload.get("ip", "")).strip()
	reason = str(payload.get("reason", "manual")).strip()
	if not ip:
		return jsonify({"status": "error", "message": "IP address is required."}), 400
	try:
		add_blocked_ip(ip, reason)
		return jsonify({"status": "success", "blocked_ip": ip})
	except Exception as exc:
		return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/blocked/<ip>", methods=["DELETE"])
@require_authentication
def api_unblock(ip: str):
	try:
		ip = str(ip).strip()
		if not ip:
			return jsonify({"status": "error", "message": "IP address is required."}), 400
		remove_blocked_ip(ip)
		return jsonify({"status": "success", "unblocked_ip": ip})
	except Exception as exc:
		return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/download_dataset", methods=["POST"])
@require_authentication
def api_download_dataset():
	"""Download a dataset from a provided URL into the data/ folder.

	Request payload: { "url": "https://..." }
	"""
	payload = request.get_json(silent=True) or {}
	url = str(payload.get("url", "")).strip()
	if not url:
		return jsonify({"status": "error", "message": "Dataset URL is required."}), 400
	try:
		# Save to data directory using last path component
		from urllib.request import urlopen

		from urllib.parse import urlsplit

		parsed = urlsplit(url)
		filename = os.path.basename(parsed.path) or "dataset.csv"
		dest_path = os.path.join(PROJECT_ROOT, "data", filename)
		with urlopen(url) as resp, open(dest_path, "wb") as out:
			out.write(resp.read())
		return jsonify({"status": "success", "path": dest_path})
	except Exception as exc:
		return jsonify({"status": "error", "message": f"Download failed: {exc}"}), 500


@app.route("/api/me", methods=["GET"])
def api_me():
	"""Return the current authentication and profile state."""

	return jsonify(
		{
			"authenticated": is_authenticated(),
			"username": session.get("username"),
			"role": session.get("role", "admin"),
			"theme": get_setting("theme", DEFAULT_THEME),
			"active_engine": get_active_engine(),
			"alert_threshold": get_alert_threshold(),
		}
	)


@app.route("/api/login", methods=["POST"])
def api_login():
	"""Authenticate an admin user and start a secure session."""

	payload = request.get_json(silent=True) or request.form or {}
	username = str(payload.get("username", "")).strip()
	password = str(payload.get("password", ""))

	if not username or not password:
		return jsonify({"status": "error", "message": "Username and password are required."}), 400

	if not verify_admin_credentials(username, password):
		return jsonify({"status": "error", "message": "Invalid credentials."}), 401

	session.clear()
	session["authenticated"] = True
	session["username"] = username
	session["role"] = "admin"
	session["login_time"] = datetime.now(timezone.utc).isoformat()

	return jsonify(
		{
			"status": "success",
			"message": "Authenticated successfully.",
			"user": {
				"username": username,
				"role": "admin",
			},
		}
	)


@app.route("/api/logout", methods=["POST"])
def api_logout():
	"""End the current authenticated session."""

	session.clear()
	return jsonify({"status": "success", "message": "Logged out."})


@app.route("/api/profile", methods=["GET"])
@require_authentication
def api_profile():
	"""Return a read-only admin profile and system summary."""

	return jsonify(
		{
			"status": "success",
			"profile": {
				"username": session.get("username"),
				"role": session.get("role", "admin"),
				"login_time": session.get("login_time"),
			},
			"system": {
				"project": "Intelligent Intrusion Detection System",
				"backend": "Flask",
				"database": "SQLite forensic log",
				"models": {
					"random_forest": bool(rf_model is not None),
					"cnn_lstm": bool(cnn_lstm_model is not None),
				},
			},
		}
	)


@app.route("/api/settings", methods=["GET", "POST"])
@require_authentication
def api_settings():
	"""Get or update persisted dashboard settings."""

	if request.method == "GET":
		return jsonify(
			{
				"status": "success",
				"settings": {
					"engine": get_active_engine(),
					"theme": get_setting("theme", DEFAULT_THEME),
					"alert_threshold": get_alert_threshold(),
				},
			}
		)

	payload = request.get_json(silent=True) or {}
	try:
		updated_settings = update_runtime_settings(payload)
		return jsonify({"status": "success", "settings": updated_settings})
	except ValueError as exc:
		return jsonify({"status": "error", "message": str(exc)}), 400
	except Exception as exc:
		return jsonify({"status": "error", "message": f"Unable to update settings: {exc}"}), 500


@app.route("/api/config", methods=["GET", "POST"])
@require_authentication
def api_config():
	"""Get or update the active inference engine configuration."""

	if request.method == "GET":
		return jsonify({"status": "success", "active_engine": get_active_engine()})

	try:
		engine = get_request_engine()
		persist = request.args.get("persist", "true").strip().lower() not in {"0", "false", "no"}
		if persist:
			engine = set_active_engine(engine)
		else:
			global active_engine
			active_engine = engine
		return jsonify({"status": "success", "active_engine": engine, "persisted": persist})
	except ValueError as exc:
		return jsonify({"status": "error", "message": str(exc)}), 400
	except Exception as exc:
		return jsonify({"status": "error", "message": f"Unable to update configuration: {exc}"}), 500


@app.route("/api/predict", methods=["POST"])
@require_authentication
def api_predict():
	"""Placeholder direct prediction endpoint kept for compatibility."""

	json_payload = request.get_json(silent=True) or {}
	engine = normalize_engine(json_payload.get("engine"), fallback=get_active_engine())
	if not models_loaded:
		return jsonify({"status": "error", "message": "Models are not loaded."}), 503

	features = json_payload.get("features")
	if features is None:
		return jsonify({"status": "error", "message": "Missing 'features' payload."}), 400

	try:
		feature_array = np.asarray(features, dtype=float)
		if feature_array.ndim != 1:
			raise ValueError("The 'features' payload must be a one-dimensional numeric array.")
		predicted_index, model_used, confidence, rf_confidence = predict_row(feature_array, engine)
		classification = decode_class_index(predicted_index)
		return jsonify(
			{
				"status": "success",
				"engine": engine,
				"prediction": classification,
				"confidence": confidence,
				"model_used": model_used,
				"rf_confidence": rf_confidence,
			}
		)
	except ValueError as exc:
		return jsonify({"status": "error", "message": str(exc)}), 400
	except Exception as exc:
		return jsonify({"status": "error", "message": f"Prediction failed: {exc}"}), 500


@app.route("/api/upload_scan", methods=["POST"])
@require_authentication
def api_upload_scan():
	"""Scan an uploaded CSV file using hybrid RF/CNN-LSTM inference."""

	if not models_loaded:
		return jsonify({"status": "error", "message": "Models are not loaded."}), 503

	try:
		engine = get_request_engine()
	except ValueError as exc:
		return jsonify({"status": "error", "message": str(exc)}), 400

	if "file" not in request.files:
		return jsonify({"status": "error", "message": "No file part in request."}), 400

	uploaded_file = request.files["file"]
	if uploaded_file.filename == "":
		return jsonify({"status": "error", "message": "No file selected."}), 400
	if not uploaded_file.filename.lower().endswith(".csv"):
		return jsonify({"status": "error", "message": "Only CSV uploads are supported."}), 400

	try:
		df = pd.read_csv(uploaded_file)
		if "duration" not in df.columns:
			uploaded_file.seek(0)
			df = pd.read_csv(uploaded_file, header=None)
			kdd_cols = [
				"duration",
				"protocol_type",
				"service",
				"flag",
				"src_bytes",
				"dst_bytes",
				"land",
				"wrong_fragment",
				"urgent",
				"hot",
				"num_failed_logins",
				"logged_in",
				"num_compromised",
				"root_shell",
				"su_attempted",
				"num_root",
				"num_file_creations",
				"num_shells",
				"num_access_files",
				"num_outbound_cmds",
				"is_host_login",
				"is_guest_login",
				"count",
				"srv_count",
				"serror_rate",
				"srv_serror_rate",
				"rerror_rate",
				"srv_rerror_rate",
				"same_srv_rate",
				"diff_srv_rate",
				"srv_diff_host_rate",
				"dst_host_count",
				"dst_host_srv_count",
				"dst_host_same_srv_rate",
				"dst_host_diff_srv_rate",
				"dst_host_same_src_port_rate",
				"dst_host_srv_diff_host_rate",
				"dst_host_serror_rate",
				"dst_host_srv_serror_rate",
				"dst_host_rerror_rate",
				"dst_host_srv_rerror_rate",
				"label",
				"difficulty",
			]
			df.columns = kdd_cols[: len(df.columns)]
		if df.empty:
			return jsonify({"status": "error", "message": "Uploaded CSV is empty."}), 400
		if len(df) > MAX_UPLOAD_ROWS:
			return jsonify({"status": "error", "message": f"Upload exceeds the supported row limit of {MAX_UPLOAD_ROWS}."}), 400

		scaled_data = scale_uploaded_frame(df)

		breakdown = {label: 0 for label in CLASS_NAMES}
		batch_predictions = predict_batch(scaled_data, engine)

		for prediction in batch_predictions:
			predicted_label = decode_class_index(prediction["predicted_index"])
			breakdown[predicted_label] += 1

			# Log the prediction to the forensics database
			log_alert(
				timestamp=datetime.now(timezone.utc).isoformat(),
				source_ip=generate_fake_ip(),
				protocol="tcp",  # Generic protocol for uploaded scans
				classification=predicted_label,
				confidence=float(prediction["confidence"]),
				model_used=str(prediction["model_used"]),
			)

		return jsonify(
			{
				"status": "success",
				"total": int(len(df)),
				"engine": engine,
				"breakdown": breakdown,
			}
		)
	except Exception as exc:
		return (
			jsonify(
				{
					"status": "error",
					"message": f"Invalid or unsupported CSV format: {exc}",
				}
			),
			400,
		)


@app.route("/api/history", methods=["GET"])
@require_authentication
def api_history():
	try:
		engine = get_request_engine(default=get_active_engine())
		limit = int(request.args.get("limit", "100"))
		limit = max(1, min(limit, 1000))

		with get_db_connection() as conn:
			cursor = conn.execute(
				"""
				SELECT id, timestamp, source_ip, protocol, classification, confidence, model_used
				FROM alerts
				ORDER BY id DESC
				LIMIT ?
				""",
				(limit,),
			)
			alerts = [dict(row) for row in cursor.fetchall()]

		return jsonify(
			{
				"status": "success",
				"count": len(alerts),
				"engine": engine,
				"alerts": alerts,
			}
		)
	except ValueError as exc:
		return jsonify({"status": "error", "message": str(exc)}), 400
	except Exception as exc:
		return jsonify({"status": "error", "message": f"Error retrieving alert history: {exc}"}), 500


@app.route("/api/live_feed", methods=["GET"])
@require_authentication
def api_live_feed():
	"""Simulate a live traffic classification event using hybrid inference.

	The route randomly picks one preprocessed NSL-KDD row from the startup pool.
	The Random Forest model makes the first pass. If its confidence is high
	enough, the RF classification is accepted. Otherwise, the same row is passed
	through the CNN-LSTM model for the final decision.
	"""

	if not models_loaded or not sample_data_loaded:
		return (
			jsonify(
				{
					"status": "error",
					"message": "Models or sample data are not loaded.",
				}
			),
			503,
		)

	try:
		engine = get_request_engine()
	except ValueError as exc:
		return jsonify({"status": "error", "message": str(exc)}), 400

	# Pick a single traffic record from the live simulation pool.
	sample_index = int(np.random.randint(0, sample_scaled_features.shape[0]))
	sample_row = sample_scaled_features[sample_index]
	raw_protocol = str(sample_protocol_types[sample_index])
	predicted_class_index, model_used, confidence, _rf_confidence = predict_row(sample_row, engine)

	# Convert the encoded prediction back to the project class label.
	classification = decode_class_index(predicted_class_index)

	# Log this live feed event to the forensics database
	timestamp = datetime.now(timezone.utc).isoformat()
	source_ip = generate_fake_ip()

	# If the generated source IP is blocked, short-circuit and mark as blocked
	if is_ip_blocked(source_ip):
		classification = "Blocked"
		confidence = 1.0
		model_used = "BLOCKED"
		log_alert(
			timestamp=timestamp,
			source_ip=source_ip,
			protocol=raw_protocol,
			classification=classification,
			confidence=confidence,
			model_used=model_used,
		)
		return jsonify(
			{
				"timestamp": timestamp,
				"sourceIP": source_ip,
				"protocol": raw_protocol,
				"classification": classification,
				"confidence": confidence,
				"model_used": model_used,
				"engine": engine,
			}
		)
	log_alert(
		timestamp=timestamp,
		source_ip=source_ip,
		protocol=raw_protocol,
		classification=classification,
		confidence=confidence,
		model_used=model_used,
	)

	return jsonify(
		{
			"timestamp": timestamp,
			"sourceIP": source_ip,
			"protocol": raw_protocol,
			"classification": classification,
			"confidence": confidence,
			"model_used": model_used,
			"engine": engine,
		}
	)


if __name__ == "__main__":
	port = int(os.environ.get("PORT", 5001))
	print(f"Starting IIDS server on http://127.0.0.1:{port}")
	app.run(host="0.0.0.0", port=port, debug=False)
