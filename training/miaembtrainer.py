import os
import yaml
import json
import logging
import time
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn

from dataloader.miaspanishcorpora import MiaSpanishCorpora
from model.miacbowemb import MiaEmbeddingsModel
from configs.mia import (
    MIA_VOCABULARY_MINWORD_FREQUENCY,
    MIA_CONTEXT_LENGTH,
    MIA_MAXSEQUENCE_LENGTH
)

from torch.optim.lr_scheduler import LambdaLR

from torchtext.data import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer
from functools import partial
from torch.utils.data import DataLoader


def collateFunction(batch: any, text_pipeline: any) -> tuple:
    """
    This collate function is invoked by dataloader in order to process a batch,
    batch is expected to be list of text paragrahs and text pipeline is a function
    that is applied to such `batch`. Based on the cbow algorithm the context is represented
    as N past words and N future words, here we are representing N as the MIA_CONTEXT_LENGTH
    value defined as a global constant. Also, it is important to mention that this collate
    function will truncate those paragraphs longer than a threshold defined by MIA_MAXSEQUENCE_LENGTH
    constant.
    """

    # Each item in batch_input is MIA_CONTEXT_LENGTH*2 context words, in another words this represent the context (x's independent variables).
    # Each item in batch_output is a word in the center of the context, in another words this represent the dependent variable to predict (y's).

    batch_input, batch_output = [], []

    # For each paragraphs received from data loader
    for text in batch:
        text_tokens_ids = text_pipeline(text) # Apply the text pipeline function on the text

        if len(text_tokens_ids) < MIA_CONTEXT_LENGTH * 2 + 1:
            continue

        if MIA_MAXSEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MIA_MAXSEQUENCE_LENGTH]

        for idx in range(len(text_tokens_ids) - MIA_CONTEXT_LENGTH * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + MIA_CONTEXT_LENGTH * 2 + 1)]
            output = token_id_sequence.pop(MIA_CONTEXT_LENGTH)
            input_ = token_id_sequence
            batch_input.append(input_)
            batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)

    return batch_input, batch_output

class MiaEmbeddingsTrainer:
    """
    Encapsulate the mia embedding training process, this class is invoked from main script to train mia embeddings based on a spanish corpus and
    the cbow architecture for model. This trainer reads the provided configuration file in order to define its training behavior. 
    """
    def __init__(self, config_filepath: str) -> None:
        """
        Constructor, initialize mia embeddings trainer based on the provided configuration file.

        Parameters:
            config_filepath (str): Configuration filepath (path + filename) from where the trainer will take its configuration values
        Returns:
            None    
        """
        self.logger = logging.getLogger(os.getenv("GMAI_LOGDEF", "development") + "." + __name__)
        self.loss = {"train": [], "val": []} # Keep track both training and validation loss during training !!!

        self.logger.info(f"Loading mia embeddings trainer configuration from {config_filepath} ...")
        with open(config_filepath, 'r') as cf:
            self.config = yaml.safe_load(cf)        

    def train(self):
        """
        Prepare and execute the training process.
        """

        self.logger.info(f"Creating model output folder ...")
        os.makedirs(self.config["miaemb_model_dir"], exist_ok=True)

        self.logger.info(f"Creating vocabulary and training data loader ...")
        # Creates the data loader for training data
        trainDataloader, vocabulary = self.createVocabAndLoader('train', self.config["miaemb_train_batch_size"], vocab=None)

        self.logger.info(f"Creating validation data loader ...")
        # Creates the data loader for validation data
        validDataloader, _ = self.createVocabAndLoader('valid', self.config["miaemb_valid_batch_size"], vocabulary)

        self.logger.info(f"Defining model architecture and creating an instance of mia embeddings model to be trained based on cbow ...")
        # Creates the model
        miaEmbModel = MiaEmbeddingsModel(len(vocabulary.get_stoi()))

        self.logger.info(f"Creating loss function, model optimizer and learning rate strategy ...")
        lossFunction = nn.CrossEntropyLoss()
        optimizer = optim.Adam(miaEmbModel.parameters(), lr=self.config["miaemb_learning_rate"])
        learningRateScheduler = self.createLearningRateScheduler(optimizer, self.config["miaemb_epochs"], verbose=True)

        self.logger.info(f"Checking if a gpu is availabe to assign training task to it ...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        # Executes the training loop once it finish we will have our embeddings ready to save !!!
        self.logger.info(f"Execute training loop using previously created object, model, training dataloader, validation dataloader, loss function, optimizer, learning rate strategy and gpu device if available ...")
        miaEmbModel = miaEmbModel.to(device)
        self.executeTraining(miaEmbModel, trainDataloader, validDataloader, lossFunction, optimizer, learningRateScheduler, device)

        # Save all the final generated artifact results of the model training (model, loss, configuration and vocabulary).
        self.logger.info(f"Model training finished. Saving computed embeddings for trained model, loss, configuration and training vocabulary as final results ...")

        self.saveFinalTrainedModel(miaEmbModel)
        self.saveLoss()
        self.saveTrainingConfig(self.config)
        self.saveTrainingVocabulary(vocabulary)

        self.logger.info("Mia embeddgins model trained and ready to be used for inference !!!")

    def executeTraining(self, model: MiaEmbeddingsModel, trainLoader: DataLoader, validLoader: DataLoader, lossFunction: any, optimizer: any, lrScheduler: any, trainDevice: any):
        """
        Execute the model's training loop based on the number of configured epochs in trainer.

        Parameters:
            model (MiaEmbeddingsModel): The embeddings model based on cbow architecture.
            trainLoader (DataLoader): The train data loader.
            validLoader (DataLoader): The valid data loader.
            lossFunction (any): The model's loss function.
            optimizer (any): The model's optimizer.
            lrScheduler (any): A learning rate scheduler that modify such value during training.
            trainDevice (any): Either a cpu or gpu device if cude is available.
        """
        epochs = self.config["miaemb_epochs"]

        t1d = time.time()
        self.logger.info(f"Training model for {epochs} epochs")
        self.logger.info(f"Model training started: {t1d}")

        for epoch in range(epochs):
            self._trainEpoch(model, trainLoader, lossFunction, optimizer, trainDevice)
            self._validateEpoch(model, validLoader, lossFunction, trainDevice)
            self.logger.info(
                "Epoch: {}/{}, Train Loss={:.5f}, Val Loss={:.5f}".format(
                    epoch + 1,
                    epochs,
                    self.loss["train"][-1],
                    self.loss["val"][-1],
                )
            )

            lrScheduler.step()

            if self.config["miaemb_checkpoint_frequency"]:
                self._saveCheckpoint(epoch, model)

        t2d = time.time()
        self.logger.info(f"Model training finished: {t2d}")
        self.logger.info(f"Model training total duration: {str((t2d-t1d))}")        


    def _trainEpoch(self, model: MiaEmbeddingsModel, trainLoader: DataLoader, lossFunction: any, optimizer: optim.Adam, device: any) -> None:
        """
        Train model during one epoch.

        Parameters:
            model (MiaEmbeddingsModel): Model to be trained based on cbow architecture for embeddings.
            trainLoader (DataLoader): Train data loader, responsible to load training data.
            lossFunction (any): The model's loss function to be applied.
            optimizer (any): The model's optimizer to be applied during training.
            device (any): Either cpu or gpu if cuda is available.
        """
        model.train()
        running_loss = []

        for i, batch_data in enumerate(trainLoader, 1):
            inputs = batch_data[0].to(device)
            labels = batch_data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = lossFunction(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())

            if i == self.config["miaemb_train_steps"]:
                break

        epoch_loss = np.mean(running_loss)
        self.loss["train"].append(epoch_loss)

    def _validateEpoch(self, model: MiaEmbeddingsModel, validLoader: DataLoader, lossFunction: any, device: any) -> None:
        """
        Validates one epoch of training.

        Parameters:
            model (MiaEmbeddingsModel): Model to be trained based on cbow architecture for embeddings.
            validLoader (DataLoader): Train data loader, responsible to load training data.
            lossFunction (any): The model's loss function to be applied.
            device (any): Either cpu or gpu if cuda is available.
        """
        model.eval()
        running_loss = []

        with torch.no_grad():
            for i, batch_data in enumerate(validLoader, 1):
                inputs = batch_data[0].to(device)
                labels = batch_data[1].to(device)

                outputs = model(inputs)
                loss = lossFunction(outputs, labels)

                running_loss.append(loss.item())

                if i == self.config["miaemb_val_steps"]:
                    break

        epoch_loss = np.mean(running_loss)
        self.loss["val"].append(epoch_loss)


    def saveFinalTrainedModel(self, model: MiaEmbeddingsModel) -> None:
        """
        Saves the trained model in the model directory.

        Parameters:
            model (MiaEmbeddingsModel): The model to be saved.

        Returns:
            None    
        """
        modelPath = os.path.join(self.config["miaemb_model_dir"], "model.pt")
        self.logger.info(f"Saving trained model in: {modelPath}")
        torch.save(model, modelPath)


    def saveLoss(self) -> None:
        """
        Saves the computed losses during training both validation and training loss as json file in the model's directory.
        """
        lossPath = os.path.join(self.config["miaemb_model_dir"], "loss.json")
        self.logger.info(f"Saving model losses in: {lossPath}")
        with open(lossPath, "w") as fp:
            json.dump(self.loss, fp)


    def saveTrainingConfig(self, config: dict) -> None:
        """
        Saves the applied configuration during training in the model's directory.

        Parameters:
            config (dict): Configuration dictionary.
        """
        config_path = os.path.join(config["miaemb_model_dir"], "config.yaml")
        self.logger.info(f"Saving trained model configuration parameters in: {config_path}")
        with open(config_path, "w") as cf:
            yaml.dump(config, cf)
        
        
    def saveTrainingVocabulary(self, vocab: Vocab) -> None:
        """
        Saves the computed vocabulary for model's training.

        Parameters:
            vocab (Vocab): The computed vocabulary to save.

        Returns:
            None    
        """
        vocab_path = os.path.join(self.config["miaemb_model_dir"], "vocab.pt")
        self.logger.info(f"Saving trained model vocabulary in: {vocab_path}")
        torch.save(vocab, vocab_path)

    def _saveCheckpoint(self, epoch: int, model: MiaEmbeddingsModel) -> None:
        """
        Save model checkpoint to configurated model directory

        Parameters:
            epoch (int): Epoch to be saved.
            model (MiaEmbeddingsModel): Model to be saved as a checkpoint.
        Returns:
            None.    
        """
        epoch_num = epoch + 1

        if epoch_num % self.config["miaemb_checkpoint_frequency"] == 0:
            model_path = "checkpoint_{}.pt".format(str(epoch_num).zfill(3))
            model_path = os.path.join(self.config["miaemb_model_dir"], model_path)
            self.logger.info(f"Saving trained model checkpoint in: {model_path}")
            torch.save(model, model_path)

    def createLearningRateScheduler(self, optimizer, totalEpochs: int, verbose: bool = True):
        """
        Scheduler to linearly decrease learning rate, so that learning rate after the last epoch is 0.
        """
        lr_lambda = lambda epoch: (totalEpochs - epoch) / totalEpochs
        learningRateScheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=verbose)

        return learningRateScheduler



    def createVocabAndLoader(self, datasetType: str, batchSize: int, vocab: Vocab) -> tuple:
        """
        Creates both a vocabulary for the provided dataset type (train or valid) and a dataloader for it.

        Parameters:
            datasetType (str): Either 'train' or 'valid'
            batchSize (int): Self explanatory, batch size for dataloader.
            vocab (Vocab): A pytorch vocabulary object, when None is provided it will compute such vocabular otherwise use the provided one.

        Returns:
            A tuple of a configured dataloader and the computed vocabulary.    
        """

        #Creates an iterator for spanish corpora dataset
        self.logger.info(f"Converting miaspanishcorpora dataset to a map style in order dataloader can use it to created batches for traiing purpose ...")
        spanishCorporaIterator = MiaSpanishCorpora(root_data_dir=self.config['miaemb_data_dir'], split_type=datasetType)
        spanishCorporaIterator = to_map_style_dataset(spanishCorporaIterator)

        #Creates a basic tokenizer - PLEASE REVIEW ME a robust one should be used instead
        self.logger.info(f"Creating a basic tokenizer, this tokenizer will only split the text read ...")
        tokenizer = get_tokenizer(None, language="en") # This only split the line of text language is ignored !!!

        if not vocab:
            self.logger.info(f"Building vocabulary from spanish corpora iterator ...")
            vocab = self.buildVocabFromSpanishCorporaIterator(spanishCorporaIterator, tokenizer)

        # text pipeline is a lambda function that basically returns a list of ids taken from vocabulary based on the output of tokenizer
        # for instance, x = "hola soy un alien que vengo por ti" -> tokenizer will return ["hola", "soy", "un", "alien", "que", "vengo", "por", "ti"]
        # vocab will then return the list of ids for each token in such list [10, 20, 3, 7, 9, 89, 11, 24]
        textPipeLine = lambda x: vocab(tokenizer(x))

        self.logger.info(f"Creating data loader using spanish corpora iterator, batchsize = {batchSize}, suffle and collate function ...")
        dataloader = DataLoader(
            spanishCorporaIterator,
            batch_size=batchSize,
            shuffle=self.config["miaemb_shuffle"],
            collate_fn=partial(collateFunction, text_pipeline=textPipeLine),
        )
   
        self.logger.info("Returning created dataloader and computed vocabulary as a tuple ...")
        return dataloader, vocab            

    def buildVocabFromSpanishCorporaIterator(self, spanishCorporaIterator: any, tokenizer: any) -> Vocab:
        """
        Builds vocabulary from spanish corpora iterator taking advantage of pytorch function build_vocab_from_iterator

        Parameters:
            spanishCorporaIterator (any): An interator for mia spanish corpora
            tokenizer (any): A tokenizer function.

        Returns:
            A Vocab object that represent the computed vocabulary based on the spanish corpora iterated through the provided iterator.    
        """
   
        self.logger.info(f"Invoking pytorch function build_vocab_from_iterator to build vocabulary for mieembeddings model, vocabulary minimal frequency to be considered is = {MIA_VOCABULARY_MINWORD_FREQUENCY}")
        vocab = build_vocab_from_iterator(
            map(tokenizer, spanishCorporaIterator),
            specials=["<unk>"],
            min_freq=MIA_VOCABULARY_MINWORD_FREQUENCY,
        )

        vocab.set_default_index(vocab["<unk>"])

        self.logger.info("Returning vocabulary ...")
        return vocab

        


