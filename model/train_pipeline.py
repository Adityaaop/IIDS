"""Training pipeline for the Intelligent Intrusion Detection System.

This module loads NSL-KDD training data, preprocesses it for both a classical
machine learning baseline and a deep learning hybrid model, trains the models,
and persists the fitted artifacts.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, balanced_accuracy_score, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Conv1D, Dense, Dropout, LSTM, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "KDDTrain+_20Percent.txt"
SCALER_PATH = BASE_DIR / "scaler.pkl"
RF_MODEL_PATH = BASE_DIR / "rf_model.pkl"
CNN_LSTM_MODEL_PATH = BASE_DIR / "cnn_lstm_model.h5"
RF_CONFUSION_MATRIX_PATH = BASE_DIR / "rf_confusion_matrix.png"
CNN_LSTM_CONFUSION_MATRIX_PATH = BASE_DIR / "cnn_lstm_confusion_matrix.png"
ROC_CURVE_PATH = BASE_DIR / "roc_curve.png"


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


def load_data() -> pd.DataFrame:
	"""Load the NSL-KDD training dataset with the standard column layout."""

	try:
		data = pd.read_csv(DATA_PATH, header=None, names=NSL_KDD_COLUMNS)
	except FileNotFoundError as exc:
		raise FileNotFoundError(
			f"Dataset not found at {DATA_PATH}. Verify the NSL-KDD file is present."
		) from exc
	except pd.errors.ParserError as exc:
		raise ValueError(f"Failed to parse dataset at {DATA_PATH}: {exc}") from exc

	return data.drop(columns=["difficulty"])


def preprocess_data(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, StandardScaler, LabelEncoder]:
	"""Transform the raw NSL-KDD data into model-ready arrays."""

	processed = data.copy()
	processed["label"] = processed["label"].astype(str).str.strip().str.lower().map(ATTACK_MAP)

	if processed["label"].isna().any():
		unknown_labels = sorted(data.loc[processed["label"].isna(), "label"].astype(str).unique())
		raise ValueError(f"Unmapped NSL-KDD attack labels found: {unknown_labels}")

	features = processed.drop(columns=["label"])
	target = processed["label"]

	target_encoder = LabelEncoder()
	encoded_target = target_encoder.fit_transform(target)

	categorical_columns = ["protocol_type", "service", "flag"]
	features = pd.get_dummies(features, columns=categorical_columns, drop_first=False)

	scaler = StandardScaler()
	scaled_features = scaler.fit_transform(features)
	joblib.dump(scaler, SCALER_PATH)

	return scaled_features, encoded_target, scaler, target_encoder


def save_confusion_matrix_plot(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	class_names: list[str],
	output_path: Path,
	title: str,
) -> None:
	"""Save a labeled confusion matrix figure for reporting."""

	matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
	plt.figure(figsize=(8, 6))
	sns.heatmap(
		matrix,
		annot=True,
		fmt="d",
		cmap="Blues",
		xticklabels=class_names,
		yticklabels=class_names,
	)
	plt.xlabel("Predicted Label")
	plt.ylabel("True Label")
	plt.title(title)
	plt.tight_layout()
	plt.savefig(output_path, dpi=300)
	plt.close()


def save_roc_curve_plot(
	y_true: np.ndarray,
	y_score: np.ndarray,
	class_names: list[str],
	output_path: Path,
	title: str,
) -> None:
	"""Save a multiclass ROC curve plot using one-vs-rest evaluation."""

	y_true_binarized = label_binarize(y_true, classes=list(range(len(class_names))))
	if y_true_binarized.shape[1] != len(class_names):
		raise ValueError("ROC plot requires all classes to be represented in the evaluation labels.")

	plt.figure(figsize=(9, 7))
	for class_index, class_name in enumerate(class_names):
		fpr, tpr, _ = roc_curve(y_true_binarized[:, class_index], y_score[:, class_index])
		roc_auc = auc(fpr, tpr)
		plt.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC = {roc_auc:.4f})")

	micro_fpr, micro_tpr, _ = roc_curve(y_true_binarized.ravel(), y_score.ravel())
	micro_auc = auc(micro_fpr, micro_tpr)
	plt.plot(micro_fpr, micro_tpr, linestyle="--", color="black", lw=2, label=f"Micro-average (AUC = {micro_auc:.4f})")
	plt.plot([0, 1], [0, 1], linestyle=":", color="gray", lw=1.5)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.title(title)
	plt.legend(loc="lower right", fontsize=9)
	plt.tight_layout()
	plt.savefig(output_path, dpi=300)
	plt.close()


def encode_feature_frame(features: pd.DataFrame, reference_columns: list[str] | None = None) -> pd.DataFrame:
	"""One-hot encode NSL-KDD categorical fields and align the result to a schema."""

	encoded = pd.get_dummies(features, columns=["protocol_type", "service", "flag"], drop_first=False)
	if reference_columns is not None:
		encoded = encoded.reindex(columns=reference_columns, fill_value=0)
	return encoded


def train_rf(
	x_train: np.ndarray,
	x_test: np.ndarray,
	y_train: np.ndarray,
	y_test: np.ndarray,
	class_names: list[str],
) -> RandomForestClassifier:
	"""Train and evaluate the Random Forest baseline model."""

	rf_model = RandomForestClassifier(
		n_estimators=500,
		random_state=42,
		n_jobs=-1,
		class_weight="balanced_subsample",
		max_features="sqrt",
		min_samples_leaf=1,
	)
	rf_model.fit(x_train, y_train)

	predictions = rf_model.predict(x_test)
	accuracy = accuracy_score(y_test, predictions)
	balanced_accuracy = balanced_accuracy_score(y_test, predictions)
	print(f"Random Forest Accuracy: {accuracy:.4f}")
	print(f"Random Forest Balanced Accuracy: {balanced_accuracy:.4f}")
	print(classification_report(y_test, predictions, labels=list(range(len(class_names))), target_names=class_names, digits=4))
	save_confusion_matrix_plot(y_test, predictions, class_names, RF_CONFUSION_MATRIX_PATH, "Random Forest Confusion Matrix")

	joblib.dump(rf_model, RF_MODEL_PATH)
	return rf_model


def reshape_for_dl(features: np.ndarray) -> np.ndarray:
	"""Reshape 2D tabular input into the 3D tensor expected by Conv1D/LSTM."""

	if features.ndim != 2:
		raise ValueError(f"Expected a 2D array for reshaping, received shape {features.shape}.")

	return features.reshape(features.shape[0], features.shape[1], 1)


def build_cnn_lstm_model(input_shape: tuple[int, int]) -> Sequential:
	"""Build the CNN-LSTM architecture for hybrid sequence learning."""

	model = Sequential(
		[
			Conv1D(filters=128, kernel_size=3, padding="same", activation="relu", input_shape=input_shape),
			BatchNormalization(),
			Dropout(0.2),
			Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"),
			BatchNormalization(),
			MaxPooling1D(pool_size=2),
			LSTM(64, return_sequences=False),
			Dropout(0.3),
			Dense(64, activation="relu"),
			Dropout(0.2),
			Dense(5, activation="softmax"),
		]
	)
	model.compile(
		optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
		loss="sparse_categorical_crossentropy",
		metrics=["accuracy"],
	)
	return model


def train_cnnlstm(
	x_train: np.ndarray,
	x_validation: np.ndarray,
	x_test: np.ndarray,
	y_train: np.ndarray,
	y_validation: np.ndarray,
	y_test: np.ndarray,
	class_names: list[str],
) -> Sequential:
	"""Train and save the CNN-LSTM deep learning model."""

	x_train_reshaped = reshape_for_dl(x_train)
	x_validation_reshaped = reshape_for_dl(x_validation)
	x_test_reshaped = reshape_for_dl(x_test)

	class_weights_array = compute_class_weight(
		class_weight="balanced",
		classes=np.unique(y_train),
		y=y_train,
	)
	class_weights = {int(class_index): float(weight) for class_index, weight in zip(np.unique(y_train), class_weights_array)}

	model = build_cnn_lstm_model((x_train_reshaped.shape[1], x_train_reshaped.shape[2]))

	early_stop = EarlyStopping(
		monitor="val_loss",
		patience=15,
		min_delta=0.0005,
		restore_best_weights=True,
		start_from_epoch=10,
	)
	reduce_lr = ReduceLROnPlateau(
		monitor="val_loss",
		factor=0.5,
		patience=5,
		min_lr=1e-6,
		verbose=1,
	)
	checkpoint = ModelCheckpoint(
		CNN_LSTM_MODEL_PATH,
		save_best_only=True,
		monitor="val_accuracy",
		mode="max",
		verbose=1,
	)

	model.fit(
		x_train_reshaped,
		y_train,
		epochs=200,
		batch_size=128,
		validation_data=(x_validation_reshaped, y_validation),
		callbacks=[early_stop, reduce_lr, checkpoint],
		class_weight=class_weights,
		verbose=1,
	)

	evaluation_loss, evaluation_accuracy = model.evaluate(x_test_reshaped, y_test, verbose=0)
	prediction_probabilities = model.predict(x_test_reshaped, verbose=0)
	predictions = np.argmax(prediction_probabilities, axis=1)
	print(f"CNN-LSTM Accuracy: {evaluation_accuracy:.4f}")
	print(f"CNN-LSTM Loss: {evaluation_loss:.4f}")
	print(classification_report(y_test, predictions, labels=list(range(len(class_names))), target_names=class_names, digits=4))
	save_confusion_matrix_plot(y_test, predictions, class_names, CNN_LSTM_CONFUSION_MATRIX_PATH, "CNN-LSTM Confusion Matrix")
	save_roc_curve_plot(y_test, prediction_probabilities, class_names, ROC_CURVE_PATH, "CNN-LSTM ROC Curve")

	model.save(CNN_LSTM_MODEL_PATH)
	return model


def run_pipeline() -> None:
	"""Run the full ML training pipeline end to end."""

	raw_data = load_data()
	processed = raw_data.copy()
	processed["label"] = processed["label"].astype(str).str.strip().str.lower().map(ATTACK_MAP)

	if processed["label"].isna().any():
		unknown_labels = sorted(raw_data.loc[processed["label"].isna(), "label"].astype(str).unique())
		raise ValueError(f"Unmapped NSL-KDD attack labels found: {unknown_labels}")

	features = processed.drop(columns=["label"])
	target = processed["label"]

	train_features, test_features, y_train, y_test = train_test_split(
		features,
		target,
		test_size=0.2,
		random_state=42,
		stratify=target,
	)
	train_features, validation_features, y_train, y_validation = train_test_split(
		train_features,
		y_train,
		test_size=0.125,
		random_state=42,
		stratify=y_train,
	)

	train_encoded = encode_feature_frame(train_features)
	validation_encoded = encode_feature_frame(validation_features, reference_columns=list(train_encoded.columns))
	test_encoded = encode_feature_frame(test_features, reference_columns=list(train_encoded.columns))

	scaler = StandardScaler()
	x_train = scaler.fit_transform(train_encoded)
	x_validation = scaler.transform(validation_encoded)
	x_test = scaler.transform(test_encoded)
	joblib.dump(scaler, SCALER_PATH)

	target_encoder = LabelEncoder()
	y_train_encoded = target_encoder.fit_transform(y_train)
	y_validation_encoded = target_encoder.transform(y_validation)
	y_test_encoded = target_encoder.transform(y_test)
	class_names = list(target_encoder.classes_)

	print("Training Random Forest baseline...")
	train_rf(x_train, x_test, y_train_encoded, y_test_encoded, class_names)

	print("Training CNN-LSTM hybrid model...")
	train_cnnlstm(x_train, x_validation, x_test, y_train_encoded, y_validation_encoded, y_test_encoded, class_names)

	print("Pipeline completed successfully.")


if __name__ == "__main__":
	run_pipeline()
