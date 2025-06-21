from tensorflow.keras.models import Model
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
import librosa
from data2 import test_audio_to_spectrograms, extract_mfcc_patches
from train_utils import get_train_data
import time
from tensorflow.keras.utils import to_categorical
import tensorflow as tf



def is_similar(test_sample, mapping, saved_embeddings_path, threshold=0.95, layer_index=-3, embedding_models=None):
    # Load saved embeddings
    saved_data = np.load(saved_embeddings_path)
    
    # Get test embedding
    test_embedding = embedding_models[layer_index].predict(test_sample)
    test_embedding = np.mean(test_embedding, axis=0)
    
    max_similarity = 0
    max_key = -1

    for key in saved_data.files:
        saved_embedding = saved_data[key]
        
        # Compute cosine similarity
        similarity = cosine_similarity(test_embedding.reshape(1, -1), saved_embedding.reshape(1, -1))[0][0]
        print(f"Key: {mapping[int(key)]}, Similarity: {similarity}")
        if (max_similarity < similarity):
            max_similarity = similarity
            max_key = int(key)
            
    if max_similarity >= threshold:
        return max_key, max_similarity
    else:
        return -1, max_similarity

if __name__ == '__main__':
    mapping = {
        0: "gon",
        1: "tim",
        2: "eli",
        3: "yeo",
        -1: "unknown"
    }

    model = load_model('./timZalo/cnn_model')
    layer_names = ['conv2d', 'maxpool2d', 'dropout', 'conv2d_1', 'maxpool2d_1', 'dropout_1', 'conv2d_2', 'max_pooling2d_2', 'dropout_2', 'flatten', 'dense', 'dropout_3', 'dense_1']
    outputs = [model.get_layer(name).output for name in layer_names if name in [l.name for l in model.layers]]
    embedding_models = [Model(inputs=model.input, outputs=output) for output in outputs]

    for i in range(1, 6):
        path = f"./ElizabethRealTest/audio_{i}.wav"
        signal, sr = librosa.load(path, sr=48000)
        num_classes = 4

        # Preprocess the data
        # X_test = extract_mfcc_patches(signal, sr=48000, n_mfcc=128, frames_per_spectrogram=128)
        # X_test = np.expand_dims(X_test, axis=-1)

        X_test = extract_mfcc_patches(
            signal, sr=48000,
            n_mfcc=13,
            frame_length=2048,
            hop_length=1024,
            frames_per_patch=128
        )
        X_test = np.expand_dims(X_test, axis=-1)            

        test_result, similarity = is_similar(X_test, mapping, "./real_embeddings/train_embeddings.npz", layer_index=-3, embedding_models=embedding_models)

        print(f"prediction:{mapping[test_result]}, simlarity:{similarity}")


