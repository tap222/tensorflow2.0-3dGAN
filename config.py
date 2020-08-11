LATENT_DEPTH = 200

BATCH_SIZE = 128
NUM_EPOCHS = 1

MODEL_SAVE_DIR = "./learned-models/3dGAN-tf2"
MODEL_NAME = "3dGAN"

OBJ = 'chair'
HYPARAMS = {
    "chair": {
        "project_shape": [2,2,2,256],
        "gen_filters_list": [ 256, 128, 64,32,1],
        "disc_filters_list": [32,64, 128, 256]
    }
}
DATA_PATH = '/content/data/train/chair'
