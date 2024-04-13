import os

from torchdata.datapipes.iter import FileOpener, FileLister
from torchtext._internal.module_utils import is_module_available

def MiaSpanishCorpora(root_data_dir: str, split_type: str):
    """Spanish Corpora Dataset

    Number of lines: 300904000 (300M)
    Number of tokens: 2996016962 (3B)
    Number of chars: 18431160978 (18.4B)

    Parameters:
        root_data_dir: Directory where the spanish corpora documents are saved.
        split_type: Split or Splits to be returned. Can be a string or tuple of strings.

    Returns:
        Datapipe that produces text from spanish corpora dataset.    
    """
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError(
            "Package 'torchdata' not found. Please install following instructions at 'https://github.com/pytorch/data'"
        )

    #What is happenning here is following
      
    # 1.- List all the text files (spanish corpora dataset) present in provided root data directory taking into account the split type either (train, valid or test)
    # 2.- Open each text file and retursn text file path and stream as a tuple
    # 3.- Read lines for each openned file, for each line in the stream, yields a tuple of file name and the line
    mia_corpora_dp = FileLister(root= os.path.join(root_data_dir, split_type), recursive=True).filter(lambda fname: fname.endswith('.txt'))
    mia_corpora_dp = FileOpener(mia_corpora_dp, mode="b")

    return mia_corpora_dp.readlines(strip_newline=False, decode=True, return_path=False)