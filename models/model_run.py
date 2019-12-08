from models.autoencoder_env_model import AutoEncoderEnvironment
from models.vae_env_model import VAE
from models.model_train import get_dataset
import numpy as np
import matplotlib.pyplot as plt

number = 571
# model = AutoEncoderEnvironment()
model = VAE(256)

path_weights = '/Users/dgrebenyuk/Research/dataset/weights'
path_val = '/Users/dgrebenyuk/Research/dataset/validation.tfrecord'

model.load_weights(['en', 'de'], path_weights)

data = get_dataset(path_val)

for i, (states, actions, labels) in enumerate(data.take(number)):
    if i == (number - 1):
        # z = model.encode(states[np.newaxis, ...])
        mean, logvar = model.infer(states[np.newaxis, ...])
        z = model.reparameterize(mean, logvar)
        x_pred = model.decode(z, apply_sigmoid=True)
        # x_pred = model.sample()

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
