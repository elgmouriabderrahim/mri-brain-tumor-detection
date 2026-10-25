from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = 224


def build_model(backbone):
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    if backbone == "mobilenetv2":
        base = keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                              input_shape=(IMG_SIZE, IMG_SIZE, 3))
        x = inputs                                   # our [-1,1] input is what MobileNetV2 expects
    else:
        base = keras.applications.ResNet50(include_top=False, weights="imagenet",
                                           input_shape=(IMG_SIZE, IMG_SIZE, 3))
        # ResNet50 has its own preprocessing, so convert from [-1,1]
        x = layers.Lambda(lambda t: keras.applications.resnet50.preprocess_input((t + 1.0) * 127.5))(inputs)

    base.trainable = False                           # freeze the pretrained layers
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(4, activation="softmax")(x)   # head for the 4 classes
    model = keras.Model(inputs, outputs)
    compile_model(model, 1e-3)
    return model, base


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def unfreeze_top(model, base, n_layers=30, lr=1e-5):
    """Fine-tuning: unfreeze the last n layers of the pretrained model."""
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    compile_model(model, lr)                         # recompile after changing trainable


def run_experiment(backbone, train_ds, val_ds, epochs=5):
    model, base = build_model(backbone)

    model.fit(train_ds, validation_data=val_ds, epochs=epochs)       # phase 1: train the new head
    head_acc = model.evaluate(val_ds, verbose=0)[1]

    unfreeze_top(model, base)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)       # phase 2: fine-tuning
    ft_acc = model.evaluate(val_ds, verbose=0)[1]

    return {"model": backbone, "val_acc_frozen": head_acc, "val_acc_finetuned": ft_acc}