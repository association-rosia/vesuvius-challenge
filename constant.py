from os.path import join, exists
from os import pardir

KAGGLE_FRAGMENTS_PATH = join(pardir, pardir, 'input', 'vesuvius-challenge-ink-detection', 'train')
COOKIECUTTER_FRAGMENTS_PATH = join('data', 'raw', 'train')
PARENT_FRAGMENTS_PATH = join(pardir, pardir, 'data', 'raw', 'train')
NOTEBOOK_FRAGMENTS_PATH = join(pardir, 'data', 'raw', 'train')
FRAGMENTS_PATH = KAGGLE_FRAGMENTS_PATH if exists(KAGGLE_FRAGMENTS_PATH) \
                                       else COOKIECUTTER_FRAGMENTS_PATH if exists(COOKIECUTTER_FRAGMENTS_PATH) \
                                       else PARENT_FRAGMENTS_PATH if exists(PARENT_FRAGMENTS_PATH) \
                                       else NOTEBOOK_FRAGMENTS_PATH

KAGGLE_WORKING_DIR = join(pardir, pardir, 'working')
COOKIECUTTER_MODELS_DIR = 'models'
MODELS_DIR = KAGGLE_WORKING_DIR if exists(KAGGLE_WORKING_DIR) else COOKIECUTTER_MODELS_DIR

KAGGLE_TEST_FRAGMENTS = join(pardir, pardir, 'input', 'vesuvius-challenge-ink-detection', 'test')
COOKIECUTTER_TEST_FRAGMENTS = join('data', 'raw', 'test')
TEST_FRAGMENTS = KAGGLE_TEST_FRAGMENTS if exists(KAGGLE_TEST_FRAGMENTS) else COOKIECUTTER_TEST_FRAGMENTS

KAGGLE_WORKING_DIR = join(pardir, pardir, 'working')
COOKIECUTTER_SUBMISSIONS_PATH = join('submissions')
TEST_FRAGMENTS = KAGGLE_WORKING_DIR if exists(KAGGLE_WORKING_DIR) else COOKIECUTTER_SUBMISSIONS_PATH

TRAIN_FRAGMENTS = ['1', '2']
VAL_FRAGMENTS = ['3']

Z_START = 27
Z_DIM = 8
TILE_SIZE = 256
TILE_SIZE_x2 = TILE_SIZE * 2
TILE_SIZE_x3 = TILE_SIZE * 3

###############################