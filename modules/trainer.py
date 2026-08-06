import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
from modules.transform import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    transformed_name,
)


def input_fn(file_pattern, tf_transform_output, batch_size=32):
    # Membaca data TFRecord terkompresi
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


def build_model_from_hyperparameters(hyperparameters, tf_transform_output):
    # Membuat model berdasarkan hyperparameter terpilih dari komponen Tuner
    hp_dict = hyperparameters.get("values", {})

    input_features = []
    for feature in NUMERICAL_FEATURES:
        input_features.append(
            tf.keras.Input(shape=(1,), name=transformed_name(feature), dtype=tf.float32)
        )
    for feature in CATEGORICAL_FEATURES:
        input_features.append(
            tf.keras.Input(shape=(1,), name=transformed_name(feature), dtype=tf.int64)
        )

    concat_inputs = tf.keras.layers.concatenate(input_features)
    x = concat_inputs

    num_layers = hp_dict.get("num_layers", 2)
    for i in range(num_layers):
        units = hp_dict.get(f"units_{i}", 64)
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        dropout_rate = hp_dict.get(f"dropout_{i}", 0.2)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=input_features, outputs=outputs)

    learning_rate = hp_dict.get("learning_rate", 1e-3)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    # Menambahkan layer transformasi tft ke model untuk digunakan saat serving
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY, None)

        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)

        outputs = model(transformed_features)
        return {"outputs": outputs}

    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs):
    # Callback function utama yang dieksekusi oleh komponen Trainer TFX
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output, batch_size=32)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, batch_size=32)

    # Memuat hyperparameter terbaik dari Tuner
    hparams = fn_args.hyperparameters or {}
    model = build_model_from_hyperparameters(hparams, tf_transform_output)

    # Pelatihan model
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=10,
    )

    # Menyimpan model dengan signature serving untuk TF Serving
    signatures = {
        "serving_default": _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        )
    }

    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)
