from glob import glob

import wandb

from Fiberdata.fiberdata import FiberData, SlidingWindowDataset
from Fiberdata.tdt import TDTFile
from Models.RNNModelsTorch import LSTMAutoencoder, train_model, load_model

hyperparameter_defaults = dict(
    epochs=20000,
    window_size=1000,
    step_size=1000,
    compression_size=50,
    learning_rate=1e-5,
    dropout=0.0,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    batch_size=2 ** 8,
    plot_frequency=100,
    save_frequency=10000,
    save_path='models',
    patience=1000,
    cooldown=5,
    factor=0.9,
)

run = wandb.init(project="brainwaveAnalysis-GRU", entity="bugsiesegal", config=hyperparameter_defaults)

config = wandb.config

data = []

for path in glob("/home/bugsie/PycharmProjects/brainwaveAnalysis2/data/D2-vehicle-fiberdata/*/"):
    data.append(FiberData(TDTFile(path), "LMag", 0))

slidingdata = SlidingWindowDataset(data[0], config["window_size"], config["step_size"])

# model = load_model("/home/bugsie/PycharmProjects/brainwaveAnalysis2/models/model_epoch_10000.pth",
#                    config["window_size"],
#                    [config["window_size"] * 2, config["window_size"]],
#                    config["compression_size"]
#                    )

model = LSTMAutoencoder(config["window_size"], [config["window_size"], int(config["window_size"]/4)],
                       config["compression_size"], config["dropout"])

wandb.watch(model, log_freq=config["plot_frequency"])

train_model(model, slidingdata, config["epochs"], config["batch_size"],
            config["learning_rate"], plot_frequency=config["plot_frequency"], save_frequency=config["save_frequency"],
            save_path=config["save_path"], beta1=config["beta1"], beta2=config["beta2"], epsilon=config["epsilon"],
            patience=config["patience"], cooldown=config["cooldown"], factor=config["factor"],
            tensorboard_active=False, )

wandb.finish()
