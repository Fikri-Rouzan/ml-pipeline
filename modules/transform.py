import tensorflow as tf
import tensorflow_transform as tft

# Definisi nama fitur
NUMERICAL_FEATURES = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "MaxHR",
    "Oldpeak",
]

CATEGORICAL_FEATURES = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope",
]

LABEL_KEY = "HeartDisease"


def transformed_name(key: str) -> str:
    # Helper untuk memberikan suffix _xf pada nama fitur terkonversi
    return key + "_xf"


def preprocessing_fn(inputs):
    # Callback function yang dieksekusi oleh komponen Transform TFX
    outputs = {}

    # Transformasi fitur numerik
    for feature in NUMERICAL_FEATURES:
        numeric_val = tf.cast(inputs[feature], tf.float32)
        outputs[transformed_name(feature)] = tft.scale_to_z_score(numeric_val)

    # Transformasi fitur kategorikal
    for feature in CATEGORICAL_FEATURES:
        cat_val = inputs[feature]
        if cat_val.dtype in [tf.int64, tf.int32]:
            cat_val = tf.strings.as_string(cat_val)
        outputs[transformed_name(feature)] = tft.compute_and_apply_vocabulary(cat_val)

    # Transformasi label utama
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
