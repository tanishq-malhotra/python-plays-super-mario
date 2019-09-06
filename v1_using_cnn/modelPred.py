from keras.models import load_model

model = load_model('cnn.model')

def predict(frame):
    pred = model.predict(frame)
    return pred