import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, BatchNormalization
from keras.layers import MaxPooling2D, Activation, Flatten


class DQNModel:
    """Class containing the deep q network model"""

    def __init__(self, H=250, W=160, C=3):
        """creates an instance of the dqn model

        Args:
            H: int, height of the input image
            W: int, width of the input image
            C: int, number of channels of the input image
        """
        self.height = H
        self.width = W
        self.channels = C

        self.model = self.build()
        self.model = self.compile(self.model)

    def build(self):
        """builds the model

        Returns: keras.model.Model, the keras representation of the model
        """
        # inputs = Input(shape=(self.height, self.width, self.channels))
        model = Sequential()
        model.add(Conv2D(filters=40, kernel_size=4, strides=(
            2, 2), padding="same", input_shape=(self.height, self.width, self.channels)))
        model.add(Activation("relu"))
        model.add(Conv2D(filters=40, kernel_size=4,
                         strides=(2, 2), padding="same"))

        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=40, kernel_size=4,
                         strides=(2, 2), padding="same"))
        model.add(Activation("relu"))
        model.add(Flatten())

        model.add(Dense(512))
        model.add(Activation("relu"))

        model.add(Dense(3))
        model.add(Activation("softmax"))
        return model

    def compile(self, model, optimizer="adam", lr=0.001):
        """compiles the model"""
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
        return model

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def fit_generator(self, *args, **kwargs):
        return self.model.fit_generator(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def plot_model(self, name="model.png", shapes=True, layers=True):
        keras.utils.plot_model(self.model, to_file=name, show_shapes=shapes, show_layer_names=layers)

if __name__ == '__main__':
    import gym
    import numpy as np
    env = gym.make("Skiing-v0")
    obs = env.reset()
    obs = np.expand_dims(obs, 0)
    model = DQNModel()
    model.plot_model()
    print(model.predict(obs))
