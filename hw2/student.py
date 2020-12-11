#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

import torch
import re
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import numpy as np
import string as s
import re as r
# import sklearn

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

# str -> [str]
def tokenise(sample):
  """
  Called before any processing of the text has occurred.
  """
  processed = sample.split()
  return processed

# [str] ->
def preprocessing(sample):
  """
  Called after tokenising but before numericalising.
  """
  sample = [r.sub(r'[^\x00-\x7f]', r'', w) for w in sample]
  # remove punctuations
  sample = [w.strip(s.punctuation) for w in sample]
  # remove numbers
  sample = [r.sub(r'[0-9]', r'', w) for w in sample]
  return sample


def postprocessing(batch, vocab):
  """
  Called after numericalising but before vectorising.
  """
  wordfreq = vocab.freqs
  total = 0
  for v in wordfreq.values():
      total += v
  avgfreq = total / len(wordfreq)
  worditos = vocab.itos
  for (i, x) in enumerate(batch):
      for (j, y) in enumerate(x):
          if 2 >= wordfreq[worditos[y]]:
              x[j] = 0
  return batch

stopWords = ['i', 'me', 'my', 'myself', 'we', 'our','you', "you're", "you've", 'your', 'yourself', 'yourselves', 'he', 'him', 'his', 'she', 'her', 'it', "it's", 'its',  'they', 'them', 'their', 'what', 'which', 'who', 'this', 'that',  'these', 'am', 'is', 'are', 'was', 'were', 'be','been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the','and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'into', 'through', 'before', 'after', 'above', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off','over', 'under', 'again', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all','any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only','own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just',"don't", 'should', 'now', "couldn't","didn't", 'doesn', "doesn't",  "hadn't",  "hasn't", "haven't","isn't","shouldn't","wasn't", 'weren', "weren't","won't", "wouldn't"]

VectorSize = 50
wordVectors = GloVe(name='6B', dim=VectorSize)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
  """
  Your model will be assessed on the predictions it makes, which must be in
  the same format as the dataset ratings and business categories.  The
  predictions must be of type LongTensor, taking the values 0 or 1 for the
  rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
  outputs a different representation convert the output here.
  """
  ratingOutput = torch.round(tnn.Sigmoid()(ratingOutput)).long()
  categoryOutput = torch.argmax(categoryOutput, axis=1)
  # OLD
  # ratingOutput = tnn.Sigmoid()(ratingOutput).long()
  # categoryOutput = np.argmax(categoryOutput, axis=1)  # Won't work with cuda:0
  return ratingOutput, categoryOutput

################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
  """
  Class for creating the neural network.  The input to your network will be a
  batch of reviews (in word vector form).  As reviews will have different
  numbers of words in them, padding has been added to the end of the reviews
  so we can form a batch of reviews of equal length.  Your forward method
  should return an output for both the rating angerd the business category.
  """

  def __init__(self, hiddenSize, numLayers, vectorSize):
    super(network, self).__init__()

    self.dropout = tnn.Dropout(0.5)
    self.hiddenSize = hiddenSize
    self.numLayers = numLayers
    self.vectorSize = vectorSize
    self.lin1 = tnn.Linear(self.hiddenSize*self.numLayers, 64)

    # Rating network
    self.initRatingModel()
    # Category network
    self.initCategoryModel()

  def initRatingModel(self):
    self.ratingLSTM = tnn.LSTM(input_size=self.vectorSize,
      hidden_size=self.hiddenSize,
      batch_first=True,
      num_layers=self.numLayers,
      bias=True,
      dropout=0.5,
      bidirectional=True
      )
    self.ratingLin2 = tnn.Linear(64, 1)

  def initCategoryModel(self):
    self.categoryLSTM = tnn.LSTM(input_size=self.vectorSize,
      hidden_size=self.hiddenSize,
      batch_first=True,
      num_layers=self.numLayers,
      bias=True,
      dropout=0.5,
      bidirectional=True)
    self.categoryLin2 = tnn.Linear(64, 5)

  # (batch_size, max(length), vector_size), tensor(32)
  # output -> [batch_size, 1], (batch_size, 5]
  def forward(self, input, length):
    # LSTM
    output1, _ = self.ratingLSTM(input)

    # Linear
    output1 = self.lin1(output1[:, -1])
    output1 = self.ratingLin2(output1)

    # LSTM
    output2, _ = self.categoryLSTM(input)

    # Linear
    output2 = self.lin1(output2[:, -1])
    output2 = self.categoryLin2(output2)

    return output1.squeeze(), output2.squeeze(),  

class loss(tnn.Module):
  """
  Class for creating the loss function.  The labels and outputs from your
  network will be passed to the forward method during training.
  """

  def __init__(self):
    super(loss, self).__init__()

  def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
    ratingOutput = ratingOutput.float()
    ratingTarget = ratingTarget.float()
    ratingLoss = tnn.BCEWithLogitsLoss()(ratingOutput, ratingTarget)
    categoryLoss = tnn.CrossEntropyLoss()(categoryOutput, categoryTarget)
    loss = ratingLoss + categoryLoss
    return loss

hiddenSize = 256      # number of hidden neurons
numLayers = 2         # number of layers
net = network(hiddenSize, numLayers, VectorSize)
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.8
batchSize = 96 # 32
epochs = 27
# optimiser = toptim.SGD(net.parameters(), lr=lr)
optimiser = toptim.Adam(net.parameters(), lr=0.001)