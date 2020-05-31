# ----------- Train ----------

LEARNING_RATE = 0.001
TRAIN_TEST_SPLIT = 0.2
BATCH_SIZE = 256
NB_EPOCHS = 15

# -------- Validation / Inference ----------
TEST_BATCH_SIZE = 1
# Stride of the sliding window in inference
OFFSET_STEP = 10
TILE_SIZE = 20


SEED = 14
dataset_path = '../planesnet/planesnet.json'
name_model = "/Binary_plane_classifier.h5"

# Inference
raster_path = "../../Classification_Planet_planes/Planet_airports/Rendered_Hong_Kong_2020_May.tif"
pred_name = "../../Classification_Planet_planes/Planet_airports/Prediction_Hong_Kong_2020_May_stratified_model.tif"