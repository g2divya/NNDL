
# ==========================================
# 1. MODEL CONFIGURATION
# ==========================================

from pathlib import Path
import zipfile

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

MODEL_DIRS = [
    BASE_DIR / "CREMA-CNN-BILSTM-ATTENTION-RESULTS",
    PROJECT_ROOT / "CREMA-CNN-BILSTM-ATTENTION-RESULTS",
    BASE_DIR,
    PROJECT_ROOT / "models",
    PROJECT_ROOT
]

PREFERRED_NAMES = [
    "best_crema_bilstm_attention.keras",
    "best_crema_bilstm.keras",
    "best_model.keras",
    "final_model.keras"
]


# ==========================================
# 2. FIND KERAS MODEL
# ==========================================

def find_model():

    model_files = []

    for directory in MODEL_DIRS:

        if directory.exists() and directory.is_dir():

            model_files.extend(
                directory.glob("*.keras")
            )

    # Remove duplicate paths
    model_files = list(dict.fromkeys(model_files))

    if not model_files:
        checked = "\n".join(
            str(directory) for directory in MODEL_DIRS
        )

        raise FileNotFoundError(
            "No .keras model found.\n"
            "Directories checked:\n" + checked
        )

    # Select preferred model
    for name in PREFERRED_NAMES:

        for file in model_files:

            if file.name == name:
                return file

    return model_files[0]


# ==========================================
# 3. VALIDATE MODEL FILE
# ==========================================

def validate_model_file(model_path):

    if model_path.stat().st_size == 0:
        raise ValueError("The model file is empty.")

    if not zipfile.is_zipfile(model_path):

        with open(model_path, "rb") as file:
            content = file.read(300)

        if b"git-lfs.github.com/spec/v1" in content:
            raise ValueError(
                "Git LFS pointer found instead of actual model weights."
            )

        if content.lstrip().startswith(
            (b"import ", b"from ", b"#", b"class ", b"def ")
        ):
            raise ValueError(
                "This file contains Python source code, not a trained "
                "Keras model. Save the trained model using model.save()."
            )

        raise ValueError(
            "Invalid or corrupted Keras model archive."
        )

    with zipfile.ZipFile(model_path, "r") as archive:

        if archive.testzip() is not None:
            raise ValueError("The model archive is corrupted.")

        files = archive.namelist()

        if "config.json" not in files:
            raise ValueError("config.json is missing from the model.")

        if "metadata.json" not in files:
            raise ValueError("metadata.json is missing from the model.")


# ==========================================
# 4. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    selected_model = find_model()

    validate_model_file(selected_model)

    st.info(
        f"Loading model: {selected_model.name}"
    )

    try:

        model = tf.keras.models.load_model(
            selected_model,
            custom_objects={
                "FrequencyPooling": FrequencyPooling,
                "CREMA>FrequencyPooling": FrequencyPooling
            },
            compile=False,
            safe_mode=True
        )

    except Exception as error:

        raise RuntimeError(
            f"Failed to load model: {selected_model}\n"
            f"Original error: {str(error)}"
        ) from error

    if model.output_shape[-1] != 6:
        raise ValueError(
            f"Expected 6 output classes, got {model.output_shape[-1]}."
        )

    return model
