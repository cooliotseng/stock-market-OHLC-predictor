from pathlib import Path
import numpy as np
from fastai.vision.augment import aug_transforms
from fastai.vision.core import imagenet_stats
from fastai.vision.data import ImageDataLoaders
from pathlib import Path
import pandas as pd
import numpy as np
from fastai.vision.core import imagenet_stats
from fastai.vision.data import ImageDataLoaders

bs = 64
# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart

path=Path('Candle Data')
path_save=Path('Candle Data/Processed')
path.ls()
np.random.seed(42)



np.random.seed(42)
data = ImageDataLoaders.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=aug_transforms(flip_vert=False, max_lighting=0.1, max_zoom=1.05, max_warp=0.,max_rotate=3), size=224, num_workers=4)
data.normalize(imagenet_stats)
