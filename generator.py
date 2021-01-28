from codes.FCVAE import FCVAE
from codes.utils import *

learning_rate = 1e-3
num_epoch = 5
step = 100

model = FCVAE(activation="sigmoid")
model.compile(optimizer="adam", metrics=["mae"])
model.build((None, 400, 600, 3))
model.summary()

loader = DataLoader()

x, y = loader.getTrainData(batch_size=1)
model.fit(x, y, epochs=3)
