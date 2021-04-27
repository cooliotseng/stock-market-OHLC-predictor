import  fastai
from fastai.vision import *
from fastai.metrics import error_rate
import numpy as np

bs = 64
# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart
path=Path('data/Candle Sticks/Candle Data')
path_save=Path('data/Candle Sticks/Processed')
print(path.ls())
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(flip_vert=False, max_lighting=0.1, max_zoom=1.05, max_warp=0.,max_rotate=3), size=224, num_workers=4).normalize(imagenet_stats)
print(data.classes)

learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(10, max_lr=slice(1e-6,1e-4))
learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(5, max_lr=slice(1e-6,1e-4))


learn.lr_find()

learn.recorder.plot()

learn.fit_one_cycle(5,max_lr=slice(1e-5,1e-4))
learn.save('First Model')
