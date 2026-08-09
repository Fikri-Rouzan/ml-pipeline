import tensorflow as tf
import tensorflow_transform as tft
import keras_tuner as kt
from typing import NamedTuple, Dict, Text, Any
from tfx.components.trainer.fn_args_utils import FnArgs
from modules.transform import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    transformed_name,
)

TunerFnResult = NamedTuple(
    "TunerFnResult", [("tuner", kt.Tuner), ("fit_kwargs", Dict[Text, Any])]
)


def build_model(
    hp: kt.HyperParameters, tf_transform_output: tft.TFTransformOutput
) -> tf.keras.Model:
    # Membuat arsitektur Keras model dengan hyperparameter
    input_features = []
    encoded_features = []

    # Input layer untuk fitur numerik
    for feature in NUMERICAL_FEATURES:
        inp = tf.keras.Input(
            shape=(1,), name=transformed_name(feature), dtype=tf.float32
        )
        input_features.append(inp)
        encoded_features.append(inp)

    # Input layer untuk fitur kategorikal
    for feature in CATEGORICAL_FEATURES:
        inp = tf.keras.Input(shape=(1,), name=transformed_name(feature), dtype=tf.int64)
        input_features.append(inp)
        encoded_features.append(tf.cast(inp, tf.float32))

    # Penggabungan seluruh input
    concat_inputs = tf.keras.layers.concatenate(encoded_features)
    x = concat_inputs

    # Dynamic hidden layers
    num_layers = hp.Int("num_layers", min_value=1, max_value=3, step=1)
    for i in range(num_layers):
        units = hp.Int(f"units_{i}", min_value=32, max_value=128, step=32)
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        dropout_rate = hp.Float(f"dropout_{i}", min_value=0.1, max_value=0.4, step=0.1)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Output layer
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=input_features, outputs=outputs)

    learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def input_fn(file_pattern, tf_transform_output, batch_size=32):
    # Membaca dan membuat dataset TensorFlow dari file TFRecord
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=lambda filenames: tf.data.TFRecordDataset(
            filenames, compression_type="GZIP"
        ),
        label_key=transformed_name(LABEL_KEY),
    )
    return dataset


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    # Callback function utama untuk komponen Tuner TFX
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output, batch_size=32)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, batch_size=32)

    tuner = kt.RandomSearch(
        hypermodel=lambda hp: build_model(hp, tf_transform_output),
        objective=kt.Objective("val_accuracy", direction="max"),
        max_trials=5,
        directory=fn_args.working_dir,
        project_name="heart_failure_tuning",
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_dataset,
            "validation_data": eval_dataset,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
        },
    )
