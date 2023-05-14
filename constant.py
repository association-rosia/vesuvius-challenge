from os.path import join, exists
from os import pardir

###### KAGGLE DIRECTORIES ######

KAGGLE_WORKING_DIR = join(pardir, pardir, 'working')
KAGGLE_FRAGMENTS_PATH = join(pardir, pardir, 'input', 'vesuvius-challenge-ink-detection', 'train')
KAGGLE_TEST_FRAGMENTS_PATH = join(pardir, pardir, 'input', 'vesuvius-challenge-ink-detection', 'test')

################################

### COOKIECUTTER DIRECTORIES ###

COOKIECUTTER_MODELS_DIR = 'models'
COOKIECUTTER_SUBMISSIONS_DIR = 'submissions'
COOKIECUTTER_FRAGMENTS_PATH = join('data', 'raw', 'train')
COOKIECUTTER_TEST_FRAGMENTS_PATH = join('data', 'raw', 'test')

################################

#### NOTEBOOKS DIRECTORIES ####

PARENT_FRAGMENTS_PATH = join(pardir, pardir, 'data', 'raw', 'train')
NOTEBOOK_FRAGMENTS_PATH = join(pardir, 'data', 'raw', 'train')
NOTEBOOK_TEST_FRAGMENTS_PATH = join(pardir, 'data', 'raw', 'train')

###############################

######### DIRECTORIES #########

MODELS_DIR = KAGGLE_WORKING_DIR if exists(KAGGLE_WORKING_DIR) else COOKIECUTTER_MODELS_DIR

SUBMISSION_DIR = KAGGLE_WORKING_DIR if exists(KAGGLE_WORKING_DIR) else COOKIECUTTER_SUBMISSIONS_DIR

FRAGMENTS_PATH = KAGGLE_FRAGMENTS_PATH if exists(KAGGLE_FRAGMENTS_PATH) \
                                       else COOKIECUTTER_FRAGMENTS_PATH if exists(COOKIECUTTER_FRAGMENTS_PATH) \
                                       else PARENT_FRAGMENTS_PATH if exists(PARENT_FRAGMENTS_PATH) \
                                       else NOTEBOOK_FRAGMENTS_PATH

TEST_FRAGMENTS = KAGGLE_TEST_FRAGMENTS_PATH if exists(KAGGLE_TEST_FRAGMENTS_PATH) else COOKIECUTTER_TEST_FRAGMENTS_PATH

###############################

########## VARIABLES ##########

TRAIN_FRAGMENTS = ['1', '2']
VAL_FRAGMENTS = ['3']

Z_START = 27
Z_DIM = 8
TILE_SIZE = 256

###############################