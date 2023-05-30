from glob import glob

from Fiberdata.fiberdata import FiberData, SlidingWindowDataset
from Fiberdata.tdt import TDTFile
from Models.RNNModelsTorch import LSTMAutoencoder, train_model, load_model


data = []

for path in glob("/home/bugsie/PycharmProjects/brainwaveAnalysis2/data/D2-vehicle-fiberdata/*/"):
    data.append(FiberData(TDTFile(path), "LMag", 0))

slidingdata = SlidingWindowDataset(data[0], 100, 10)

model = LSTMAutoencoder(100, [200, 100], 10)

# model = load_model('/home/bugsie/PycharmProjects/brainwaveAnalysis2/notebooks/models/model_epoch_500.pth', 100, 10, 5)

train_model(model, slidingdata, 500, 2**8, 1e-4, plot_frequency=5, save_frequency=10, save_path='models')
