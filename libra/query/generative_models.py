from libra.preprocessing.image_preprocesser import (setwise_preprocessing,
                                                    csv_preprocessing,
                                                    classwise_preprocessing,
                                                    set_distinguisher,
                                                    already_processed)
from libra.query.supplementaries import generate_id
from keras import Model
from keras.models import Sequential
from keras.layers import (Dense, Conv2D, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D)

def gan(instruction=None,
        read_mode=None,
        verbose=None,
        preprocess=None,
        epochs=None,
        height=None,
        width=None):
    return
    '''return {
        'id': generate_id(),
        'data_type': read_mode,
        'data': {'train': X_train, 'test': X_test},
        'shape': input_shape,
        "model": model,
        'losses': {
            'training_loss': history.history['loss'],
            'val_loss': history.history['val_loss']},
        'accuracy': {
            'training_accuracy': history.history['accuracy'],
            'validation_accuracy': history.history['val_accuracy']},
        'num_classes': (2 if num_classes == 1 else num_classes),
        'data_sizes': {'train_size': processInfo['train_size'], 'test_size': processInfo['test_size']}}
        '''