
# ==============================================================
# NetSentinel — Network Intrusion Detection Dashboard
# RBM + Markov Random Field + Bayesian Network
# ==============================================================

import os
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# --------------------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------------------

st.set_page_config(
    page_title="NetSentinel | Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

FILES = {
    "features": "features.pkl",
    "labels": "label_names.pkl",
    "rbm": "rbm.pt",
    "rbm_threshold": "rbm_threshold.pkl",
    "mrf": "markov_network.pkl",
    "bayesian": "bayesian.pkl",
    "bn_thresholds": "disc_thresholds.pkl",
    "rbm_metrics": "rbm_metrics.json",
    "mrf_metrics": "markov_metrics.json",
    "bn_metrics": "bayesian_metrics.json",
}

DEFAULT_LABELS = {
    0: "BENIGN",
    1: "DDoS",
    2: "PortScan",
    3: "BruteForce",
    4: "DoS",
}

st.markdown("""
<style>
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 2rem;
}
.hero {
    padding: 1.3rem;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,.3);
    margin-bottom: 1rem;
}
.hero h1 {
    margin-bottom: .25rem;
}
.small-note {
    font-size: .9rem;
    opacity: .8;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🛡️ NetSentinel</h1>
<p>AI-assisted Network Intrusion Detection System</p>
<p class="small-note">
Restricted Boltzmann Machine · Markov Random Field · Bayesian Network
</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------
# 2. FILE HELPERS
# --------------------------------------------------------------

def file_path(key):
    return BASE_DIR / FILES[key]


def file_exists(key):
    return file_path(key).is_file()


def load_pickle_file(key):
    path = file_path(key)
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json_file(key):
    with open(file_path(key), "r", encoding="utf-8") as f:
        return json.load(f)


def load_torch_file(key):
    path = file_path(key)

    # Only load checkpoints that you created or trust.
    try:
        return torch.load(
            path, map_location="cpu", weights_only=False
        )
    except TypeError:
        return torch.load(path, map_location="cpu")


def as_float_threshold(value):
    if isinstance(value, dict):
        for key in ("threshold", "rbm_threshold", "value"):
            if key in value:
                value = value[key]
                break

    arr = np.asarray(value).reshape(-1)
    if len(arr) == 0:
        raise ValueError("Threshold file contains no value.")

    return float(arr[0])


# --------------------------------------------------------------
# 3. ROBUST RBM IMPLEMENTATION
# --------------------------------------------------------------

class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden, activation="sigmoid"):
        super().__init__()

        self.n_visible = int(n_visible)
        self.n_hidden = int(n_hidden)
        self.activation = str(activation).lower()

        self.W = nn.Parameter(
            torch.randn(self.n_visible, self.n_hidden) * 0.01
        )
        self.h_bias = nn.Parameter(torch.zeros(self.n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(self.n_visible))

    def activate(self, x):
        if self.activation == "relu":
            return torch.relu(x)
        if self.activation == "tanh":
            return torch.tanh(x)
        return torch.sigmoid(x)

    def hidden_probability(self, visible):
        return self.activate(visible @ self.W + self.h_bias)

    def visible_probability(self, hidden):
        return self.activate(hidden @ self.W.T + self.v_bias)

    def forward(self, visible):
        return self.visible_probability(
            self.hidden_probability(visible)
        )


def find_checkpoint_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError(
            "RBM checkpoint is not a dictionary. "
            "Inspect how rbm.pt was saved."
        )

    for key in ("model_state", "state_dict", "model_state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]

    # The checkpoint may itself be a state dictionary.
    if any(
        key in checkpoint
        for key in ("W", "weight", "weight_vh", "weight_hv")
    ):
        return checkpoint

    raise ValueError(
        "Cannot find model weights in rbm.pt. "
        f"Available keys: {list(checkpoint.keys())}"
    )


def normalize_state_dict(state):
    # Remove common wrappers from saved parameter names.
    normalized = {}

    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "model.", "rbm."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        normalized[new_key] = value

    return normalized


def load_rbm_model():
    checkpoint = load_torch_file("rbm")
    state = normalize_state_dict(find_checkpoint_state(checkpoint))

    # Support the parameter names used by the RBM class above.
    weight_key = next(
        (
            key for key in ("W", "weight", "weight_vh", "weight_hv")
            if key in state
        ),
        None,
    )

    if weight_key is None:
        raise ValueError(
            f"Could not identify RBM weight matrix. Keys: {list(state)}"
        )

    W = state[weight_key]
    if not isinstance(W, torch.Tensor) or W.ndim != 2:
        raise ValueError("RBM weight matrix must be a 2-D tensor.")

    # For the W layout used in this project: visible x hidden.
    n_visible = int(checkpoint.get("n_visible", W.shape[0]))
    n_hidden = int(checkpoint.get("n_hidden", W.shape[1]))

    if W.shape == (n_hidden, n_visible):
        W = W.T

    if W.shape != (n_visible, n_hidden):
        raise ValueError(
            "RBM checkpoint dimensions do not match the expected "
            f"architecture. Weight shape: {tuple(W.shape)}, "
            f"visible={n_visible}, hidden={n_hidden}."
        )

    activation = checkpoint.get("activation", "sigmoid")
    model = RBM(n_visible, n_hidden, activation)

    # Map common aliases to the parameters in this RBM implementation.
    converted = {}

    converted["W"] = W

    hidden_bias = next(
        (state[k] for k in ("h_bias", "hidden_bias", "hb") if k in state),
        None,
    )
    visible_bias = next(
        (state[k] for k in ("v_bias", "visible_bias", "vb") if k in state),
        None,
    )

    if hidden_bias is None or visible_bias is None:
        raise ValueError(
            "Checkpoint does not contain recognizable hidden and visible "
            f"biases. Available keys: {list(state)}"
        )

    converted["h_bias"] = hidden_bias.reshape(-1)
    converted["v_bias"] = visible_bias.reshape(-1)

    model.load_state_dict(converted, strict=True)
    model.eval()

    return model


# --------------------------------------------------------------
# 4. LOAD EACH ARTIFACT INDEPENDENTLY
# --------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def cached_load_rbm(path_string, modified_time):
    return load_rbm_model()


@st.cache_resource(show_spinner=False)
def cached_load_pickle(path_string, modified_time):
    with open(path_string, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def cached_load_json(path_string, modified_time):
    with open(path_string, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_load(key, kind="pickle"):
    path = file_path(key)

    if not path.is_file():
        return None, f"File not found: {path.name}"

    try:
        modified = path.stat().st_mtime

        if kind == "rbm":
            result = cached_load_rbm(str(path), modified)
        elif kind == "json":
            result = cached_load_json(str(path), modified)
        else:
            result = cached_load_pickle(str(path), modified)

        return result, None

    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


features, features_error = safe_load("features")
labels, labels_error = safe_load("labels")
rbm_model, rbm_error = safe_load("rbm", "rbm")
rbm_threshold_raw, threshold_error = safe_load("rbm_threshold")
mrf_bundle, mrf_error = safe_load("mrf")
bn_saved, bn_error = safe_load("bayesian")
bn_thresholds, bn_threshold_error = safe_load("bn_thresholds")

rbm_threshold = None
if rbm_threshold_raw is not None:
    try:
        rbm_threshold = as_float_threshold(rbm_threshold_raw)
    except Exception as exc:
        threshold_error = str(exc)

if features is not None:
    features = list(features)

if labels is not None:
    try:
        labels = {int(k): str(v) for k, v in labels.items()}
    except Exception:
        labels = DEFAULT_LABELS.copy()
else:
    labels = DEFAULT_LABELS.copy()

# Bayesian file may contain a model directly or a bundle.
bn_model = bn_saved
if isinstance(bn_saved, dict):
    for candidate in ("model", "bayesian_model", "bn_model"):
        if candidate in bn_saved:
            bn_model = bn_saved[candidate]
            break


# --------------------------------------------------------------
# 5. SIDEBAR AND DIAGNOSTICS
# --------------------------------------------------------------

st.sidebar.title("🧭 Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Dashboard",
        "Traffic Analysis",
        "Model Status",
        "Training Metrics",
        "About",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Artifact availability")

statuses = {
    "RBM": rbm_model is not None,
    "MRF": mrf_bundle is not None,
    "Bayesian Network": bn_model is not None,
}

for name, available in statuses.items():
    if available:
        st.sidebar.success(f"{name}: Loaded")
    else:
        st.sidebar.warning(f"{name}: Unavailable")

st.sidebar.caption(f"Artifact directory: {BASE_DIR}")


# --------------------------------------------------------------
# 6. MODEL HELPERS
# --------------------------------------------------------------

def run_rbm(X):
    if rbm_model is None:
        raise RuntimeError(rbm_error or "RBM is unavailable.")

    if rbm_threshold is None:
        raise RuntimeError(
            threshold_error or "RBM threshold is unavailable."
        )

    if X.shape[1] != rbm_model.n_visible:
        raise ValueError(
            f"RBM expects {rbm_model.n_visible} features, "
            f"but the CSV contains {X.shape[1]}."
        )

    with torch.no_grad():
        tensor = torch.as_tensor(X, dtype=torch.float32)
        reconstruction = rbm_model(tensor).cpu().numpy()

    scores = np.mean((X - reconstruction) ** 2, axis=1)
    anomalies = scores > rbm_threshold

    return scores, anomalies


def mrf_predict(discrete_df):
    if mrf_bundle is None:
        raise RuntimeError(mrf_error or "MRF model unavailable.")

    if not isinstance(mrf_bundle, dict):
        raise ValueError("Saved MRF artifact is not a model bundle.")

    factors = mrf_bundle.get("factors", [])
    nodes = mrf_bundle.get("feature_nodes", [])
    class_node = mrf_bundle.get("class_node", "AttackType")
    states = mrf_bundle.get("node_states", {})

    if not factors or not nodes:
        raise ValueError("Saved MRF factors or feature nodes are missing.")

    if class_node not in states:
        raise ValueError(f"MRF class states for {class_node} are missing.")

    missing = [n for n in nodes if n not in discrete_df.columns]
    if missing:
        raise ValueError(f"Missing MRF evidence columns: {missing}")

    class_states = states[class_node]
    index_maps = {
        node: {int(v): i for i, v in enumerate(states[node])}
        for node in nodes
    }
    class_index = {
        int(v): i for i, v in enumerate(class_states)
    }

    log_factors = [
        (
            list(factor.variables),
            np.log(np.maximum(np.asarray(factor.values), 1e-300))
        )
        for factor in factors
    ]

    cache = {}
    predictions = []
    posteriors = []

    for row in discrete_df[nodes].itertuples(index=False, name=None):
        key = tuple(int(v) for v in row)

        if key not in cache:
            observed = {}

            for node, value in zip(nodes, key):
                if value not in index_maps[node]:
                    raise ValueError(
                        f"Unexpected state {value} for MRF node {node}."
                    )
                observed[node] = index_maps[node][value]

            scores = []

            for class_value in class_states:
                assignment = dict(observed)
                assignment[class_node] = class_index[int(class_value)]

                total = 0.0
                for variables, log_values in log_factors:
                    factor_index = tuple(
                        assignment[v] for v in variables
                    )
                    total += float(log_values[factor_index])

                scores.append(total)

            scores = np.asarray(scores, dtype=np.float64)
            scores -= scores.max()
            probs = np.exp(scores)
            probs /= probs.sum()

            prediction = int(class_states[int(np.argmax(probs))])
            cache[key] = (prediction, probs)

        pred, probs = cache[key]
        predictions.append(pred)
        posteriors.append(probs)

    return np.asarray(predictions), np.asarray(posteriors)


def get_bn_threshold(node, threshold_dict):
    if not isinstance(threshold_dict, dict):
        return None

    if node in threshold_dict:
        item = threshold_dict[node]
        if isinstance(item, dict):
            item = item.get("thresholds", item.get("bins"))
        if item is not None:
            return np.asarray(item, dtype=float).reshape(-1)

    aliases = {
        "SYNRate": ["SYN Flag Count", "SYNRate"],
        "PacketRate": ["Flow Packets/s", "PacketRate"],
        "ByteRate": ["Flow Bytes/s", "ByteRate"],
        "AnomalyScore": ["AnomalyScore", "RBM"],
    }

    for alias in aliases.get(node, []):
        if alias in threshold_dict:
            item = threshold_dict[alias]
            if isinstance(item, dict):
                item = item.get("thresholds", item.get("bins"))
            if item is not None:
                return np.asarray(item, dtype=float).reshape(-1)

    return None


def bn_predict(evidence_df):
    if bn_model is None:
        raise RuntimeError(bn_error or "Bayesian model unavailable.")

    try:
        from pgmpy.inference import VariableElimination
    except Exception as exc:
        raise RuntimeError(f"Could not import pgmpy inference: {exc}")

    class_node = "AttackType"

    if class_node not in bn_model.nodes():
        raise ValueError(
            f"Bayesian model does not contain node '{class_node}'."
        )

    evidence_nodes = [
        n for n in bn_model.nodes() if n != class_node
    ]

    missing = [
        n for n in evidence_nodes if n not in evidence_df.columns
    ]
    if missing:
        raise ValueError(f"Missing Bayesian evidence columns: {missing}")

    inference = VariableElimination(bn_model)
    cache = {}
    predictions = []
    posterior_rows = []
    class_states = None

    for row in evidence_df[evidence_nodes].itertuples(
        index=False, name=None
    ):
        key = tuple(int(v) for v in row)

        if key not in cache:
            evidence = dict(zip(evidence_nodes, key))
            result = inference.query(
                variables=[class_node],
                evidence=evidence,
                show_progress=False
            )

            probabilities = np.asarray(result.values).reshape(-1)
            result_states = list(result.state_names[class_node])

            pred = int(result_states[int(np.argmax(probabilities))])
            cache[key] = (pred, probabilities, result_states)

        pred, probs, result_states = cache[key]
        predictions.append(pred)
        posterior_rows.append(probs)
        class_states = result_states

    return (
        np.asarray(predictions),
        np.asarray(posterior_rows),
        class_states,
    )


# --------------------------------------------------------------
# 7. FEATURE PREPARATION
# --------------------------------------------------------------

def numeric_feature_matrix(df):
    if features is None:
        raise RuntimeError(
            features_error or "features.pkl is unavailable."
        )

    missing = [name for name in features if name not in df.columns]
    if missing:
        raise ValueError(
            "Uploaded CSV is missing required training features: "
            + ", ".join(missing)
        )

    numeric_df = df[features].apply(pd.to_numeric, errors="coerce")

    if numeric_df.isna().any().any():
        bad_columns = numeric_df.columns[
            numeric_df.isna().any()
        ].tolist()
        raise ValueError(
            f"Non-numeric or missing values in columns: {bad_columns}"
        )

    X = numeric_df.to_numpy(dtype=np.float32)

    if not np.isfinite(X).all():
        raise ValueError("Input contains infinite values.")

    return X


def build_mrf_input(df, X, rbm_scores=None):
    if mrf_bundle is None:
        raise RuntimeError("MRF bundle unavailable.")

    nodes = mrf_bundle["feature_nodes"]
    thresholds = mrf_bundle["bin_thresholds"]

    out = pd.DataFrame(index=np.arange(len(df)))

    aliases = {
        "SYN Flag Count": ["SYN Flag Count"],
        "Flow Packets/s": ["Flow Packets/s"],
        "Flow Bytes/s": ["Flow Bytes/s"],
    }

    for node in nodes:
        if node == "AnomalyScore":
            if rbm_scores is None:
                raise ValueError("MRF requires RBM anomaly scores.")
            values = rbm_scores
        else:
            source = next(
                (
                    candidate
                    for candidate in aliases.get(node, [node])
                    if candidate in features
                ),
                None,
            )

            if source is None:
                raise ValueError(
                    f"Cannot map MRF node '{node}' to a source feature."
                )

            values = X[:, features.index(source)]

        if node not in thresholds:
            raise ValueError(f"Missing MRF thresholds for '{node}'.")

        out[node] = np.digitize(
            values, np.asarray(thresholds[node], dtype=float)
        ).astype(int)

    return out


def build_bn_input(df, X, rbm_scores=None):
    if bn_model is None:
        raise RuntimeError("Bayesian model unavailable.")

    if bn_thresholds is None:
        raise RuntimeError(
            bn_threshold_error or "Bayesian thresholds unavailable."
        )

    nodes = [n for n in bn_model.nodes() if n != "AttackType"]
    out = pd.DataFrame(index=np.arange(len(df)))

    aliases = {
        "SYNRate": ["SYNRate", "SYN Flag Count"],
        "PacketRate": ["PacketRate", "Flow Packets/s"],
        "ByteRate": ["ByteRate", "Flow Bytes/s"],
    }

    for node in nodes:
        if node == "AnomalyScore":
            if rbm_scores is None:
                raise ValueError(
                    "Bayesian model requires RBM anomaly scores."
                )
            values = rbm_scores
        else:
            source = next(
                (
                    candidate
                    for candidate in aliases.get(node, [node])
                    if candidate in features
                ),
                None,
            )

            if source is None:
                raise ValueError(
                    f"Cannot map Bayesian node '{node}' to a source feature."
                )

            values = X[:, features.index(source)]

        thresholds = get_bn_threshold(node, bn_thresholds)

        if thresholds is None:
            raise ValueError(
                f"No saved discretization thresholds found for '{node}'."
            )

        out[node] = np.digitize(values, thresholds).astype(int)

    return out


# --------------------------------------------------------------
# 8. DASHBOARD PAGE
# --------------------------------------------------------------

if page == "Dashboard":
    st.subheader("System Overview")
    st.write(
        "NetSentinel combines anomaly detection and probabilistic "
        "models to analyze preprocessed network traffic."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("RBM", "Ready" if rbm_model is not None else "Unavailable")
    c2.metric("MRF", "Ready" if mrf_bundle is not None else "Unavailable")
    c3.metric(
        "Bayesian Network",
        "Ready" if bn_model is not None else "Unavailable"
    )

    st.info(
        "Upload a CSV under Traffic Analysis to generate predictions. "
        "Input must use the same feature scaling and preprocessing "
        "as the training dataset."
    )

    st.markdown("#### Attack classes")
    st.write(", ".join(labels.values()))

    st.markdown("#### Expected input")
    st.write(
        "A CSV with the columns listed in `features.pkl`, containing "
        "the same numerical representations used during training."
    )


elif page == "Traffic Analysis":
    st.subheader("Network Traffic Analysis")

    uploaded = st.file_uploader(
        "Upload preprocessed network traffic CSV",
        type=["csv"],
        key="traffic_csv",
    )

    if uploaded is not None:
        try:
            input_df = pd.read_csv(uploaded)

            st.write("Input preview")
            st.dataframe(input_df.head(10), use_container_width=True)

            X = numeric_feature_matrix(input_df)

            st.success(
                f"Validated {len(input_df):,} records and {X.shape[1]} features."
            )

            rbm_scores = None
            rbm_anomalies = None
            mrf_predictions = None
            bn_predictions = None

            # RBM
            st.markdown("### 1. RBM Anomaly Detection")
            try:
                with st.spinner("Calculating RBM reconstruction errors..."):
                    rbm_scores, rbm_anomalies = run_rbm(X)

                a, b, c = st.columns(3)
                a.metric("Records", f"{len(X):,}")
                b.metric("Anomalies", f"{int(rbm_anomalies.sum()):,}")
                c.metric(
                    "Anomaly rate",
                    f"{rbm_anomalies.mean() * 100:.2f}%"
                )

                rbm_table = pd.DataFrame({
                    "Reconstruction_Error": rbm_scores,
                    "RBM_Status": np.where(
                        rbm_anomalies, "Anomaly", "Normal"
                    )
                })

                st.dataframe(rbm_table.head(100), use_container_width=True)

                fig, ax = plt.subplots(figsize=(9, 3.5))
                ax.hist(rbm_scores, bins=40)
                ax.axvline(
                    rbm_threshold,
                    linestyle="--",
                    label="Saved threshold"
                )
                ax.set_xlabel("Reconstruction error")
                ax.set_ylabel("Records")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

            except Exception as exc:
                st.error(f"RBM prediction failed: {exc}")

            # MRF
            st.markdown("### 2. Markov Random Field")
            try:
                with st.spinner("Calculating MRF predictions..."):
                    mrf_input = build_mrf_input(
                        input_df, X, rbm_scores
                    )
                    mrf_predictions, mrf_probs = mrf_predict(mrf_input)

                mrf_table = pd.DataFrame({
                    "MRF_Predicted_Attack": [
                        labels.get(int(v), str(v))
                        for v in mrf_predictions
                    ],
                    "MRF_Score": mrf_probs.max(axis=1),
                })

                st.dataframe(mrf_table.head(100), use_container_width=True)
                st.bar_chart(
                    mrf_table["MRF_Predicted_Attack"].value_counts()
                )

            except Exception as exc:
                st.error(f"MRF prediction failed: {exc}")

            # Bayesian Network
            st.markdown("### 3. Bayesian Network")
            try:
                with st.spinner("Calculating Bayesian predictions..."):
                    bn_input = build_bn_input(
                        input_df, X, rbm_scores
                    )
                    bn_predictions, bn_probs, bn_states = bn_predict(
                        bn_input
                    )

                bn_table = pd.DataFrame({
                    "Bayesian_Predicted_Attack": [
                        labels.get(int(v), str(v))
                        for v in bn_predictions
                    ],
                    "Bayesian_Score": bn_probs.max(axis=1),
                })

                st.dataframe(bn_table.head(100), use_container_width=True)
                st.bar_chart(
                    bn_table["Bayesian_Predicted_Attack"].value_counts()
                )

            except Exception as exc:
                st.error(f"Bayesian prediction failed: {exc}")

            # Combined results
            st.markdown("### Combined Results")
            combined = input_df.copy()

            if rbm_scores is not None:
                combined["RBM_Error"] = rbm_scores
                combined["RBM_Status"] = np.where(
                    rbm_anomalies, "Anomaly", "Normal"
                )

            if mrf_predictions is not None:
                combined["MRF_Prediction"] = [
                    labels.get(int(v), str(v))
                    for v in mrf_predictions
                ]

            if bn_predictions is not None:
                combined["Bayesian_Prediction"] = [
                    labels.get(int(v), str(v))
                    for v in bn_predictions
                ]

            if (
                mrf_predictions is not None
                and bn_predictions is not None
            ):
                agreement = np.mean(
                    mrf_predictions == bn_predictions
                )
                st.metric(
                    "MRF / Bayesian agreement",
                    f"{agreement * 100:.2f}%"
                )

            st.dataframe(combined.head(100), use_container_width=True)

            st.download_button(
                "Download prediction CSV",
                data=combined.to_csv(index=False).encode("utf-8"),
                file_name="netsentinel_predictions.csv",
                mime="text/csv",
            )

        except Exception as exc:
            st.error(f"Input validation failed: {exc}")


elif page == "Model Status":
    st.subheader("Model and Artifact Diagnostics")

    artifact_rows = []

    for key, filename in FILES.items():
        path = BASE_DIR / filename
        artifact_rows.append({
            "Artifact": filename,
            "Exists": path.is_file(),
            "Size (KB)": (
                round(path.stat().st_size / 1024, 2)
                if path.is_file() else None
            ),
        })

    st.dataframe(
        pd.DataFrame(artifact_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Loading errors")

    errors = {
        "Features": features_error,
        "Labels": labels_error,
        "RBM": rbm_error,
        "RBM threshold": threshold_error,
        "MRF": mrf_error,
        "Bayesian Network": bn_error,
        "Bayesian thresholds": bn_threshold_error,
    }

    for name, error in errors.items():
        if error:
            st.error(f"{name}: {error}")
        else:
            st.success(f"{name}: Loaded successfully")

    st.warning(
        "Do not upload untrusted pickle or PyTorch checkpoint files. "
        "Loading them can execute unsafe deserialization behavior."
    )


elif page == "Training Metrics":
    st.subheader("Saved Training Metrics")

    metric_configs = [
        ("RBM", "rbm_metrics"),
        ("MRF", "mrf_metrics"),
        ("Bayesian Network", "bn_metrics"),
    ]

    loaded_any = False

    for name, key in metric_configs:
        st.markdown(f"### {name}")

        if not file_exists(key):
            st.info(f"{FILES[key]} was not found.")
            continue

        try:
            data = load_json_file(key)
            loaded_any = True

            if "accuracy" in data:
                st.metric(
                    "Saved accuracy",
                    f"{float(data['accuracy']) * 100:.2f}%"
                )

            st.json(data)

        except Exception as exc:
            st.error(f"Could not read {FILES[key]}: {exc}")

    if not loaded_any:
        st.info(
            "No training metrics JSON files were loaded. "
            "You can still inspect the models on Model Status."
        )


elif page == "About":
    st.subheader("About NetSentinel")

    st.write(
        "NetSentinel is an academic network intrusion detection "
        "project based on CICIDS2017 and three complementary methods."
    )

    st.markdown("""
    **Restricted Boltzmann Machine (RBM)**  
    Uses reconstruction error to flag potentially unusual traffic.

    **Markov Random Field (MRF)**  
    Uses pairwise potentials between discretized traffic features
    and attack classes.

    **Bayesian Network**  
    Uses conditional probability inference over learned network
    relationships.

    **Important limitations**  
    Results depend on consistent preprocessing, feature ordering,
    and saved-model compatibility. Model agreement does not prove
    that a traffic record is malicious.
    """)

st.divider()
st.caption(
    "NetSentinel · Academic prototype · Validate models and preprocessing "
    "before operational use."
)
