# ==========================================
# 1. MODEL CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "CREMA-CNN-BILSTM-ATTENTION-RESULTS"


# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}"
        )

    # Search for all Keras models
    model_files = list(MODEL_DIR.glob("*.keras"))

    if not model_files:
        raise FileNotFoundError(
            f"No .keras model found inside {MODEL_DIR}"
        )

    # Prefer the best model if available
    preferred_names = [
        "best_crema_bilstm_attention.keras",
        "best_crema_bilstm.keras",
        "best_model.keras",
        "final_model.keras"
    ]

    selected_model = None

    for name in preferred_names:
        for file in model_files:
            if file.name == name:
                selected_model = file
                break

        if selected_model:
            break

    # Otherwise use the first available model
    if selected_model is None:
        selected_model = model_files[0]

    st.info(f"Loading model: {selected_model.name}")

    model = tf.keras.models.load_model(
        selected_model,
        custom_objects={
            "FrequencyPooling": FrequencyPooling,
            "CREMA>FrequencyPooling": FrequencyPooling
        },
        compile=False,
        safe_mode=True
    )

    return model
