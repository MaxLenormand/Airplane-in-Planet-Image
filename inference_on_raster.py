import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import rasterio as rio

import torch
from binary_classifier_model import Net
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_info = torch.cuda.get_device_properties(device=device)
print(f"Using: {device} (info: {device_info}")

# Reproducability
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# Opening the Planet 4 band image as an array and keeping the metadeta
raster = rio.open(raster_path)
meta = raster.meta
array = raster.read(out_dtype='uint8')[:3,:,:]
# Deleting the raster allows to save RAM
del raster
array = array.reshape([-1, 3, array.shape[1], array.shape[2]]).transpose([0,2,3,1])
array_img = array[0]


# ===================== INFERENCE =======================
net = Net()
net.to(device)
print(net)

path_model = os.getcwd() + name_model
net.load_state_dict(torch.load(path_model))

# Creating a 2d array to put our predictions as the window moves.
# This would be similar to a mask for a segmentation task.
overlay = np.zeros((array_img.shape[0], array_img.shape[1]))

for offset_x in tqdm(range(0, TILE_SIZE, OFFSET_STEP)):
    for offset_y in tqdm(range(0, TILE_SIZE, OFFSET_STEP)):
        for x in range(offset_x, array_img.shape[0], TILE_SIZE):
            for y in range(offset_y, array_img.shape[1], TILE_SIZE):
                img = array_img[x:x + TILE_SIZE, y:y + TILE_SIZE]
                if img.shape == (TILE_SIZE, TILE_SIZE, 3):
                    img = img / 255

                    # img to tensor:
                    torch_img = torch.from_numpy(img)
                    torch_img = torch_img.reshape(-1, 3, TILE_SIZE, TILE_SIZE)

                    with torch.no_grad():
                        input_img = torch_img.to(device, dtype=torch.float32)

                        output= net(input_img)[0]
                        pred = torch.argmax(output)

                        if pred == 1:
                            overlay[x:x + TILE_SIZE, y:y + TILE_SIZE] += 10

overlay = overlay.astype(np.uint8)
fig, axis = plt.subplots(1,2,figsize=(50, 20))
axis[0].imshow(array_img)
axis[0].imshow(overlay, alpha=0.5)
axis[1].imshow(overlay)
plt.show()

# Saving the geo-referenced predictions
meta.update({"count":1})
with rio.open(pred_name, "w", **meta) as dst:
    dst.write(overlay, 1)