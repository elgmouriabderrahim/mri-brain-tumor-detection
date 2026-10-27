from tensorflow import keras
from tensorflow.keras import layers, regularizers

IMG_SIZE = 224


def build_model(backbone, dropout=0.0, l2=0.0):
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    if backbone == "mobilenetv2":
        base = keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                              input_shape=(IMG_SIZE, IMG_SIZE, 3))
        x = inputs
    else:
        base = keras.applications.ResNet50(include_top=False, weights="imagenet",
                                           input_shape=(IMG_SIZE, IMG_SIZE, 3))
        x = layers.Lambda(lambda t: keras.applications.resnet50.preprocess_input((t + 1.0) * 127.5))(inputs)

    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)                                   # Dropout
    outputs = layers.Dense(4, activation="softmax",
                           kernel_regularizer=regularizers.l2(l2) if l2 > 0 else None)(x)  # L2
    model = keras.Model(inputs, outputs)
    compile_model(model, 1e-3)
    return model, base


def get_callbacks(ckpt_path):
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]


def run_config(name, backbone, train_ds, val_ds, dropout, l2, use_callbacks, epochs=15):
    model, base = build_model(backbone, dropout, l2)
    cbs = lambda: get_callbacks(f"../models/{name}.keras") if use_callbacks else []

    h1 = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs())   # train head
    unfreeze_top(model, base)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs())   # fine-tuning

    history = {k: h1.history[k] + h2.history[k] for k in ("loss", "val_loss", "accuracy", "val_accuracy")}
    return model, history