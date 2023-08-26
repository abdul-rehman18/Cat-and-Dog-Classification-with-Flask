import os
from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

image_size = (128, 128)

# Load the trained model
model = tf.keras.models.load_model('cat_and_dog.h5')


# Set the path to your test dataset directory
test_dataset_dir = 'dog_cat/'


# Define the  labels
labels = ['cats','dogs']

# Set the upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'static/images'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Load and preprocess the uploaded image
        image = load_img(filepath, target_size=image_size)
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array /= 255.0  # Normalize the image

        # Make a prediction
        prediction = model.predict(image_array)
        # Check if prediction probability is above a threshold
        threshold = 0.9
        predicted_color_index = np.argmax(prediction)
        predicted_color_prob = prediction[0][predicted_color_index]

        # if predicted_color_prob >= threshold:
        #     predicted_color_label = color_labels[predicted_color_index]
        # else:
        #     predicted_color_label = 'unknown'

        predicted_color_label = labels[predicted_color_index]

        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(test_dataset_dir,target_size=image_size,batch_size=1,class_mode='categorical',shuffle=False)

        # Predict classes for test images
        predictions = model.predict(test_generator)
        predicted_classes = predictions.argmax(axis=-1)
        true_classes = test_generator.classes

        # Calculate confusion matrix
        confusion = confusion_matrix(true_classes, predicted_classes)

        # Calculate F1 score
        f1 = f1_score(true_classes, predicted_classes, average='weighted')

        # Calculate accuracy
        accuracy = accuracy_score(true_classes, predicted_classes)

        # Plot the confusion matrix using Seaborn
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel("predicted labels")
        plt.ylabel("true label")
        plt.title('Confusion Matrix')

        # Save the confusion matrix plot as an image
        confusion_matrix_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'confusion_matrix.png')

        plt.savefig(confusion_matrix_img_path)



        

        

        return render_template('index.html', filename=filename, predicted_color=predicted_color_label, confusion_matrix_img=confusion_matrix_img_path,f1_score=f1, accuracy=accuracy)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
