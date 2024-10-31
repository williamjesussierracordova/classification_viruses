from flask import Flask, request, jsonify
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Embedding
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import os
from flask_cors import CORS


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Ruta y nombre del modelo guardado
MODEL_PATH = 'modelo_adn.h5'
LABEL_CLASSES_PATH = 'label_classes.npy'
DATA_PATH = 'secuencias_etiquetadas_2.csv'

# Función para codificar secuencias
def encode_sequence(seq, k=1):
    encoding = {
        'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3,
        'R': 4, 'Y': 5, 'S': 6, 'W': 7, 'K': 8, 'M': 9,
        'B': 10, 'D': 11, 'H': 12, 'V': 13, 'N': 14,
        '-': 15, '.': 15
    }
    encoded = [encoding.get(nuc.upper(), 14) for nuc in seq]
    return encoded

# Función para entrenar el modelo
def train_model():
    data = pd.read_csv(DATA_PATH)
    sequences = data['sequence'].values
    labels = data['label'].values

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    np.save(LABEL_CLASSES_PATH, le.classes_)

    # Codificar secuencias
    encoded_sequences = [encode_sequence(seq) for seq in sequences]
    max_length = max(len(seq) for seq in encoded_sequences)
    padded_sequences = pad_sequences(encoded_sequences, maxlen=max_length, padding='post', value=14)

    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels_encoded, test_size=0.3, random_state=42)

    # Definir y compilar el modelo
    model = Sequential()
    model.add(Embedding(input_dim=16, output_dim=8, input_length=max_length))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(len(np.unique(labels_encoded)), activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.3, verbose=1)

    # Guardar el modelo
    model.save(MODEL_PATH)
    print('Model trained and saved successfully.')
    return model, max_length

# Función para cargar el modelo si ya existe
def load_trained_model():
    model = load_model(MODEL_PATH)
    le = LabelEncoder()
    le.classes_ = np.load(LABEL_CLASSES_PATH, allow_pickle=True)
    max_length = model.input_shape[1]
    return model, le, max_length

# Cargar o entrenar el modelo al iniciar la API
if os.path.exists(MODEL_PATH):
    model, le, max_length = load_trained_model()
    print('Model loaded successfully.')
else:
    model, max_length = train_model()
    le = LabelEncoder()
    le.classes_ = np.load(LABEL_CLASSES_PATH, allow_pickle=True)

# Preprocesar la secuencia de ADN
def preprocess_sequence(sequence, max_length):
    encoded_sequence = encode_sequence(sequence)
    padded_sequence = pad_sequences([encoded_sequence], maxlen=max_length, padding='post', value=14)
    return padded_sequence

# Ruta para la predicción
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files and 'sequence' not in request.form:
        return jsonify({'error': 'No file or sequence provided'}), 400

    sequence = None
    # Leer desde un archivo CSV si se proporciona
    if 'file' in request.files:
        file = request.files['file']
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            if 'sequence' not in df.columns:
                return jsonify({'error': 'CSV must contain a "sequence" column'}), 400
            sequence = df['sequence'].iloc[0]  # Usar la primera fila para la predicción
        else:
            return jsonify({'error': 'Invalid file format. Only CSV is accepted.'}), 400

    # Leer directamente una secuencia de ADN si se proporciona
    if 'sequence' in request.form:
        sequence = request.form['sequence']

    if not sequence:
        return jsonify({'error': 'No valid sequence found'}), 400

    # Preprocesar la secuencia
    preprocessed_sequence = preprocess_sequence(sequence, max_length)

    # Realizar la predicción
    prediction = model.predict(preprocessed_sequence)
    predicted_class_idx = np.argmax(prediction)
    predicted_class = le.inverse_transform([predicted_class_idx])[0]
    confidence = prediction[0][predicted_class_idx]

    # Devolver la respuesta en formato JSON
    return jsonify({
        'predicted_class': predicted_class,
        'confidence': float(confidence)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
