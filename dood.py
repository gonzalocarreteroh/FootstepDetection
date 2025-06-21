from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.models import load_model
from train_utils import get_train_data
from data2 import preprocess_framed_audio_to_mfcc_spectrograms, extract_mfcc_patches
from tensorflow.keras.utils import to_categorical




if __name__ == '__main__':

    # # One-hot encode the labels
    # num_classes = 2
    # class_labels_g = to_categorical(class_labels_g, num_classes=num_classes)
    # class_labels_t = to_categorical(class_labels_t, num_classes=num_classes)

    # Preprocess the data
    raw_g, labels_g = get_train_data("./GonzaloRealTrain", label=0)
    raw_t, labels_t = get_train_data("./TimRealTrain",     label=1)
    # raw_e, labels_e = get_train_data("./samples/ElizabethDemo", label=2)
    # raw_y, labels_y = get_train_data("./samples/YeongDemo", label=3)

    # 2) Extract patches and repeat labels
    specs_g, lbls_g = [], []
    specs_t, lbls_t = [], []
    # specs_e, lbls_e = [], []
    # specs_y, lbls_y = [], []

    for sig, lbl in zip(raw_g, labels_g):
        specs = extract_mfcc_patches(
            sig, sr=48000,
            n_mfcc=13,
            frame_length=2048,
            hop_length=1024,
            frames_per_patch=128
        )
        specs_g.append(specs)
        lbls_g.append(np.full((specs.shape[0],), lbl, dtype=int))

    X_g = np.vstack(specs_g)
    y_g = np.concatenate(lbls_g)

    for sig, lbl in zip(raw_t, labels_t):
        specs = extract_mfcc_patches(
            sig, sr=48000,
            n_mfcc=13,
            frame_length=2048,
            hop_length=1024,
            frames_per_patch=128
        )
        specs_t.append(specs)
        lbls_t.append(np.full((specs.shape[0],), lbl, dtype=int))

    X_t = np.vstack(specs_t)
    y_t = np.concatenate(lbls_t)

    # for sig, lbl in zip(raw_y, labels_y):
    #     specs = extract_mfcc_patches(
    #         sig, sr=48000,
    #         n_mfcc=13,
    #         frame_length=2048,
    #         hop_length=1024,
    #         frames_per_patch=128
    #     )
    #     specs_y.append(specs)
    #     lbls_y.append(np.full((specs.shape[0],), lbl, dtype=int))

    # X_y = np.vstack(specs_y)
    # y_y = np.concatenate(lbls_y)

    # for sig, lbl in zip(raw_e, labels_e):
    #     specs = extract_mfcc_patches(
    #         sig, sr=48000,
    #         n_mfcc=13,
    #         frame_length=2048,
    #         hop_length=1024,
    #         frames_per_patch=128
    #     )
    #     specs_e.append(specs)
    #     lbls_e.append(np.full((specs.shape[0],), lbl, dtype=int))
    # X_e = np.vstack(specs_e)
    # y_e = np.concatenate(lbls_e)


    # X_g, y_g = preprocess_framed_audio_to_mfcc_spectrograms(raw_audio_data_g, class_labels_g, sr=48000, n_mfcc=128, frames_per_spectrogram=128)
    # X_t, y_t = preprocess_framed_audio_to_mfcc_spectrograms(raw_audio_data_t, class_labels_t, sr=48000, n_mfcc=128, frames_per_spectrogram=128)

    # X_train_g = np.expand_dims(X_g, axis=-1)
    # X_train_t = np.expand_dims(X_t, axis=-1)
    # Build the model
    model = load_model('./timZalo/cnn_model')

    layer_names = ['conv2d', 'maxpool2d', 'dropout', 'conv2d_1', 'maxpool2d_1', 'dropout_1', 'conv2d_2', 'max_pooling2d_2', 'dropout_2', 'flatten', 'dense', 'dropout_3', 'dense_1']
    outputs = [model.get_layer(name).output for name in layer_names if name in [l.name for l in model.layers]]
    embedding_models = [Model(inputs=model.input, outputs=output) for output in outputs]

    embeddings_g = embedding_models[-3].predict(X_g)
    embeddings_t = embedding_models[-3].predict(X_t)
    # embeddings_y = embedding_models[-3].predict(X_y)
    # embeddings_e = embedding_models[-3].predict(X_e)

    average_embedding_g = np.mean(embeddings_g, axis=0)
    average_embedding_t = np.mean(embeddings_t, axis=0)
    # average_embedding_y = np.mean(embeddings_y, axis=0)
    # average_embedding_e = np.mean(embeddings_e, axis=0)
    
    embeddings_dict = {
        "0": average_embedding_g,
        "1": average_embedding_t,
        # "2": average_embedding_e,
        # "3": average_embedding_y
    }

    # Save embeddings to a single file
    np.savez('./real_embeddings/train_embeddings.npz', **embeddings_dict)

