"""
A separate module which we can import from a second process with a function to train R models
without blocking other python threads

@author Dennis Soemers
"""


def train_r_models(CS_MODELS_R_FILEPATH, OUTPUT_FILE_MODEL_LEARNING_DATA, OUTPUT_DIR, seed):
    import rpy2.robjects as robjects
    robjects.r('set.seed({})'.format(seed))
    robjects.r('source(\"{}\")'.format(CS_MODELS_R_FILEPATH))
    robjects.r('datafilename<-\"{}\"'.format(OUTPUT_FILE_MODEL_LEARNING_DATA).replace("\\", "/"))
    savepath_string = OUTPUT_DIR.replace("\\", "/")
    if not savepath_string.endswith("/"):
        savepath_string += "/"
    robjects.r('savepath<-\"{}\"'.format(savepath_string))
    robjects.r('buildNselectCSModels(datafilename,savepath)')
