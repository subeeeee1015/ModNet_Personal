import os
import shutil
import numpy as np

# val_fg = 'val/fg'

# files = sorted(os.listdir(val_fg))
# np.random.shuffle(files)
# print(len(files))

# train_files = files[:90]
# val_files = files[90:]

# for file in train_files:
#     shutil.move(os.path.join('val', 'fg', file), os.path.join('train', 'fg'))
#     shutil.move(os.path.join('val', 'alpha', file), os.path.join('train', 'alpha'))

val_fg = 'val/fg'
train_fg = 'train/fg'

train_files = sorted(os.listdir(train_fg))
val_files = sorted(os.listdir(val_fg))

with open('train.txt', 'w') as f:
    for file in train_files:
        f.write(os.path.join(train_fg, file+'\n'))
        
with open('val.txt', 'w') as f:
    for file in val_files:
        f.write(os.path.join(val_fg, file+'\n'))
        