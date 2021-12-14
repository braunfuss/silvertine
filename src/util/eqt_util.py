from silvertine.detector.core.EqT_utils import DataGeneratorPrediction, picker, generate_arrays_from_file
from silvertine.detector.core.EqT_utils import f1, SeqSelfAttention, FeedForward, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import os

def load_eqt_model():
    model_path = os.path.dirname(os.path.abspath(__file__))+"/../detector/model/EqT_model.h5"
    model_eqt = load_model(model_path,
                           custom_objects={'SeqSelfAttention': SeqSelfAttention,
                           'FeedForward': FeedForward,
                           'LayerNormalization': LayerNormalization,
                           'f1': f1
                            })

    model_eqt.compile(loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
                      loss_weights= [0.03, 0.40, 0.58],
                      optimizer=Adam(lr=0.001),
                      metrics=[f1])
    return model_eqt
