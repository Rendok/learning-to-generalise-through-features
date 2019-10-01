from models.autoencoder_env_model import AutoEncoderEnvironment
from models.model_train import get_dataset
import numpy as np
import matplotlib.pyplot as plt

number = 501
model = AutoEncoderEnvironment()

path_weights = '/Users/dgrebenyuk/Research/dataset/weights'
path_val = '/Users/dgrebenyuk/Research/dataset/validation.tfrecord'

model.load_weights(['en', 'de'], path_weights)

data = get_dataset(path_val)

for i, (states, actions, labels) in enumerate(data.take(number)):
    if i == (number - 1):
        z = model.encode(states[np.newaxis, ...])
        x_pred = model.decode(z)

        plt.imshow(states[:, :, :3])
        plt.title('State')
        plt.axis('off')
        plt.show()
        plt.imshow(states[:, :, 3:6])
        plt.title('State')
        plt.axis('off')
        plt.show()
        plt.imshow(x_pred[0, :, :, 0:3])
        plt.title('Encoded-Decoded State')
        plt.axis('off')
        plt.show()
        plt.imshow(x_pred[0, :, :, 3:6])
        plt.title('Encoded-Decoded State')
        plt.axis('off')
        plt.show()
