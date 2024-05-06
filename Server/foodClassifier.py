import os
import os.path
import numpy as np
import pandas as pd
import ast
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers.legacy import Adam
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
print("TensorFlow version:", tf.__version__)

# Load saved model and history
model_version = "Models/MobileNetv2_1.h5"
print(f"Loading {model_version} model and history...")
model = load_model(model_version)
print("Loaded model.")


def image_processing(filepath):
    """ Create a DataFrame with the filepath and the labels of the pictures
    """
    labels = [str(filepath[i]).split("/")[-2] \
              for i in range(len(filepath))]
    filepath = pd.Series(filepath, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    # Concatenate filepaths and labels
    df = pd.concat([filepath, labels], axis=1)

    # Shuffle the DataFrame and reset index
    df = df.sample(frac=1).reset_index(drop = True)
    return df

def generate_train_test_val_data(df_path):
    train_dir = Path(df_path+'/train')
    train_filepaths = list(train_dir.glob(r'**/*.jpg'))

    test_dir = Path(df_path+'/test')
    test_filepaths = list(test_dir.glob(r'**/*.jpg'))

    val_dir = Path(df_path+'/validation')
    val_filepaths = list(test_dir.glob(r'**/*.jpg'))

    train_df = image_processing(train_filepaths)
    test_df = image_processing(test_filepaths)
    val_df = image_processing(val_filepaths)

    print('-- Training set --\n')
    print(f'Number of pictures: {train_df.shape[0]}\n')
    print(f'Number of different labels: {len(train_df.Label.unique())}\n')
    print(f'Labels: {train_df.Label.unique()}')
    print(train_df.head(5))

    df_unique = train_df.copy().drop_duplicates(subset=["Label"]).reset_index()

    # Display some pictures of the dataset
    fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(8, 8),
                            subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(df_unique.Filepath[i]))
        ax.set_title(df_unique.Label[i], fontsize = 12)
    plt.tight_layout(pad=0.5)
    plt.show()

    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )

    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )

    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0,
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_images = train_generator.flow_from_dataframe(
        dataframe=val_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0,
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )

    print("Done generating train, test, and validation images.")
    return train_images, test_images, val_images

def train(train_images, val_images, model_version):
    # Fetch pretrained MobileNetV2
    pretrained_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    pretrained_model.trainable = False
    inputs = pretrained_model.input

    x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    outputs = tf.keras.layers.Dense(36, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    history = model.fit(
        train_images,
        validation_data=val_images,
        batch_size = 32,
        epochs=10,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=2,
                restore_best_weights=True
            )
        ]
    )

    # Save model
    model.save(model_version)
    print(f"Saved model {model_version}.")

    # Save history
    with open('history.pkl', 'wb') as file:
        pickle.dump(history.history, file)
    print("Saved model history.")

    return model, history


def test(model, train_images, test_images, test_df):
    pred = model.predict(test_images)
    pred = np.argmax(pred,axis=1)

    # Map the label
    labels = (train_images.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    pred1 = [labels[k] for k in pred]
    accuracy = accuracy_score(test_df.Label, pred1)

    
    print("Test Accuracy: ", accuracy)

def classify(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Load labels
    with open('Server/food_labels.txt', 'r') as file:
        labels_str = file.read()
    labels = list(ast.literal_eval(labels_str).values())
    print("Labels:", labels)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_label = labels[predicted_index]

    return predicted_label



def evaluate(model, history, test_images):
    # Plot training and validation accuracy
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot training and validation loss
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Evaluate test accuracy and loss
    test_loss, test_accuracy = model.evaluate(test_images)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    # SkLearn report
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_images.classes
    class_labels = list(test_images.class_indices.keys())
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)

    report_file_path = "Results/classification_report.txt"
    with open(report_file_path, "w") as report_file:
        report_file.write(report)

    print("Classification report has been written to:", report_file_path)


def main():
    df_path = "Datasets/Crafted_dataset"
    model_version = 'Models/MobileNetv2_1.h5'

    # Train model
    # train_images, test_images, val_images = generate_train_test_val_data(df_path)
    # model, history = train(train_images, val_images, model_version)

    # Classify a single image
    # image_path = 'Server/tmp_img.jpg'
    # predicted_label = classify(image_path, model)
    # print("Predicted label:", predicted_label)
    
    # Evaluate model
    # evaluate(model, history, test_images)
    # print("Done evaluating ")


if __name__ == "__main__":
    main()



