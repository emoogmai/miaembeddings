import logging
import os
import torch.nn as nn

from configs.mia import MIA_EMBED_DIMENSION, MIA_EMBED_MAX_NORM


class MiaEmbeddingsModel(nn.Module):
    """
    Implementation of CBOW architecture described in paper Word2Vec https://arxiv.org/abs/1301.3781
    """
    def __init__(self, vocabulary_size: int):
        """
        Creates the mia embedding model initializing it with the two layers as defined in the word2vec paper
        for cbow architecture.

        Parameters:
            vocabulary_size (int): The size of the vocabulary.
        """
        super(MiaEmbeddingsModel, self).__init__()

        self.logger = logging.getLogger(os.getenv("GMAI_LOGDEF", "development") + "." + __name__)

        # Word2Vec is a neural network based model only with two layers, 
        # embeddings and a multiconnected layer (linear) defined below.
        # you can see pytorch embedding class as a dictionary where key is
        # the 'word' and value the list and/or array with the computed embedding
        # for such word. 
        
        self.logger.info(f"Configuring mia embeddings model with two layers (embedding and fully connected), vocabulary size = {vocabulary_size}, embedding dimesion = {MIA_EMBED_DIMENSION} and normalization = {MIA_EMBED_MAX_NORM}")
        self.embeddings = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=MIA_EMBED_DIMENSION,
            max_norm=MIA_EMBED_MAX_NORM,
        )

        self.linear = nn.Linear(
            in_features=MIA_EMBED_DIMENSION,
            out_features=vocabulary_size,
        )
        self.logger.info(f"Mia embeddings model configured based on cbow architecture !!!")

    def forward(self, input_word_ids):
        """
        Execute a pass forward of the neural network inputs through all the sub-layers. Implements the computation to be
        performed at every call.

        Parameters:
            input_word_ids: The word ids (normally the index position of a word in a sequence after tokenization), for instance,
                if a your input sequence is "embeddings son la piedra angular" then this method will receive a list with the ids
                of each word based in the tokenization algorithm being used [1, 50, 25, 10, 5].
        """

        # Returns a matrix (tensor) of N x MIA_EMBED_DIMENSION where N = length of input_word_ids
        # Embeddings pytorch class is a simple lookup table where key is the 'word id' and value is
        # a list of embeddings (initialized ramdomly at the very beggingin) of MIA_EMBED_DIMENSION,
        # example:
        #
        # [2.345, 0.8908, ..., 4.2980, 6.8935]
        # [0.205, 0.2894, ..., 4.2980, 6.8935]
        # [2.345, 0.8908, ..., 4.2980, 6.8935]
        # [2.345, 0.8908, ..., 4.2980, 6.8935]
        # [2.345, 0.8908, ..., 4.2980, 6.8935]
        x = self.embeddings(input_word_ids) 

        # Computes the mean of each column value in matrix and return a matrix of 1 x MIA_EMBED_DIMENSION (flattern) where each item
        # is the mean of correspondend column
        x = x.mean(axis=1)

        # Computes the output embeddings using multiconnected linear layer and return the resultant matrix
        x = self.linear(x)
        
        return x