from os.path import join, exists
from os import pardir

###### KAGGLE DIRECTORIES ######

KAGGLE_WORKING_DIR = join(pardir, pardir, 'working')
KAGGLE_TRAIN_FRAGMENTS_PATH = join(pardir, pardir, 'input', 'vesuvius-challenge-ink-detection', 'train')
KAGGLE_TEST_FRAGMENTS_PATH = join(pardir, pardir, 'input', 'vesuvius-challenge-ink-detection', 'test')
KAGGLE_SAVE_PATH = join(pardir, pardir, 'working', 'data', 'raw')


### COOKIECUTTER DIRECTORIES ###

COOKIECUTTER_MODELS_DIR = 'models'
COOKIECUTTER_SUBMISSIONS_DIR = 'submissions'
COOKIECUTTER_TRAIN_FRAGMENTS_PATH = join('data', 'raw', 'train')
COOKIECUTTER_TEST_FRAGMENTS_PATH = join('data', 'raw', 'test')
COOKIECUTTER_TRAIN_SAVE_PATH = join('data', 'processed', 'train')
COOKIECUTTER_TEST_SAVE_PATH = join('data', 'processed', 'test')

PARENT_TRAIN_FRAGMENTS_PATH = join(pardir, pardir, 'data', 'raw', 'train')
PARENT_TEST_FRAGMENTS_PATH = join(pardir, pardir, 'data', 'raw', 'test')
PARENT_TRAIN_SAVE_PATH = join(pardir, pardir, 'data', 'processed', 'train')
PARENT_TEST_SAVE_PATH = join(pardir, pardir, 'data', 'processed', 'test')


#### NOTEBOOKS DIRECTORIES ####
NOTEBOOK_TRAIN_FRAGMENTS_PATH = join(pardir, 'data', 'raw', 'train')
NOTEBOOK_TEST_FRAGMENTS_PATH = join(pardir, 'data', 'raw', 'test')


######### DIRECTORIES #########

MODELS_DIR = KAGGLE_WORKING_DIR if exists(KAGGLE_WORKING_DIR) else COOKIECUTTER_MODELS_DIR
SUBMISSION_DIR = KAGGLE_WORKING_DIR if exists(KAGGLE_WORKING_DIR) else COOKIECUTTER_SUBMISSIONS_DIR

TRAIN_FRAGMENTS_PATH = KAGGLE_TRAIN_FRAGMENTS_PATH if exists(KAGGLE_TRAIN_FRAGMENTS_PATH) \
    else COOKIECUTTER_TRAIN_FRAGMENTS_PATH if exists(COOKIECUTTER_TRAIN_FRAGMENTS_PATH) \
    else PARENT_TRAIN_FRAGMENTS_PATH if exists(PARENT_TRAIN_FRAGMENTS_PATH) \
    else NOTEBOOK_TRAIN_FRAGMENTS_PATH

TEST_FRAGMENTS_PATH = KAGGLE_TEST_FRAGMENTS_PATH if exists(KAGGLE_TEST_FRAGMENTS_PATH) \
    else COOKIECUTTER_TEST_FRAGMENTS_PATH if exists(COOKIECUTTER_TEST_FRAGMENTS_PATH) \
    else PARENT_TEST_FRAGMENTS_PATH if exists(PARENT_TEST_FRAGMENTS_PATH) \
    else NOTEBOOK_TEST_FRAGMENTS_PATH


########## VARIABLES ##########

TRAIN_FRAGMENTS = ['1', '2', '3']
VAL_FRAGMENTS = []
TEST_FRAGMENTS = ['a', 'b', 'c']

Z_START = 0
Z_DIM = 32
TILE_SIZE = 256