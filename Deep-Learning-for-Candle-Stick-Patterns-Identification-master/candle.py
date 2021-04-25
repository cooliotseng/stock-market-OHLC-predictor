# from fastai.vision import *
# from fastai.metrics import error_rate
from pathlib import Path
import numpy as np
# from fastai.vision.core import imagenet_stats
from fastai.vision.data import ImageDataLoaders
from fastai.vision.augment import aug_transforms

from fastai.vision import *
from fastai.vision.all import *
from fastai.metrics import error_rate

bs = 64
# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart

path=Path('Candle Data')
path_save=Path('Candle Data/Processed')
path.ls()
np.random.seed(42)



np.random.seed(42)
data = ImageDataLoaders.from_folder(path, train=".ipynb_checkpoints/Training the Model and Inference-checkpoint.ipynb", valid_pct=0.2,
        ds_tfms=aug_transforms(flip_vert=False, max_lighting=0.1, max_zoom=1.05, max_warp=0.5,
                               max_rotate=3),
                                    batch_tfms=Normalize.from_stats(*imagenet_stats),
                                  size=224, num_workers=4)

learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.data = data
learn.fit_one_cycle(4)
learn.unfreeze()

learn.lr_find()


learn.recorder.plot()
