import json
import os
import random
import scipy.io
from collections import defaultdict

# modified from Karpathy's neuraltalk (https://github.com/karpathy/neuraltalk)
class BasicDataProvider:
  
  def __init__(self, dataset, **kwargs):
    print 'Initializing data provider for dataset %s...' % (dataset, )
    
    # load the dataset into memory
    dataset_root = kwargs.get('dataset_root', os.path.join('datasets', dataset))
    dataset_path = os.path.join(dataset_root, 'dataset.json')
    print 'BasicDataProvider: reading %s' % (dataset_path, )
    self.dataset = json.load(open(dataset_path, 'r'))

    # group images by their train/val/test split into a dictionary -> list structure
    self.split = defaultdict(list)
    for img in self.dataset['images']:
      self.split[img['split']].append(img)

  # "PRIVATE" FUNCTIONS
  # in future we may want to create copies here so that we don't touch the 
  # data provider class data, but for now lets do the simple thing and 
  # just return raw internal img qa pair structs. This also has the advantage
  # that the driver could store various useful caching stuff in these structs
  # and they will be returned in the future with the cache present
  def _getImage(self, img):
    """ create an image structure """
    return img

  def _getQAPair(self, qa_pair):
    """ create a QA pair structure """
    return qa_pair
  
  def _getQAMultipleChoice(self, qa_pair, shuffle = False):
    """ create a QA multiple choice structure """
    qa_pair = self._getQAPair(qa_pair)
    if 'multiple_choices' in qa_pair:
      mcs = qa_pair['multiple_choices']
      pos_idx = range(len(mcs)+1)
      # random shuffle the positions of multiple choices
      if shuffle: random.shuffle(pos_idx)
      qa_pair['mc_candidates'] = []
      for idx, k in enumerate(pos_idx):
        if k == 0:
          qa_pair['mc_candidates'].append(qa_pair['answer'])
          qa_pair['mc_selection'] = idx # record the position of the true answer
        else:
          qa_pair['mc_candidates'].append(mcs[k-1])
    return qa_pair

  # PUBLIC FUNCTIONS  
  def getSplitSize(self, split, ofwhat = 'qa_pairs'):
    """ return size of a split, either number of QA pairs or number of images """
    if ofwhat == 'qa_pairs': 
      return sum(len(img['qa_pairs']) for img in self.split[split])
    else: # assume images
      return len(self.split[split])

  def sampleImageQAPair(self, split = 'train'):
    """ sample image QA pair from a split """
    images = self.split[split]

    img = random.choice(images)
    pair = random.choice(img['qa_pairs'])

    out = {}
    out['image'] = self._getImage(img)
    out['qa_pair'] = self._getQAPair(pair)
    return out

  def sampleImageQAMultipleChoice(self, split = 'train', shuffle = False):
    """ sample image and a multiple-choice test from a split """
    images = self.split[split]

    img = random.choice(images)
    pair = random.choice(img['qa_pairs'])

    out = {}
    out['image'] = self._getImage(img)
    out['mc'] = self._getQAMultipleChoice(pair, shuffle)
    return out

  def iterImageQAPair(self, split = 'train', max_images = -1):
    for i, img in enumerate(self.split[split]):
      if max_images >= 0 and i >= max_images: break
      for pair in img['qa_pairs']:
        out = {}
        out['image'] = self._getImage(img)
        out['qa_pair'] = self._getQAPair(pair)
        yield out

  def iterImageQAMultipleChoice(self, split = 'train', max_images = -1, max_batch_size = 100, shuffle = False):
    for i, img in enumerate(self.split[split]):
      if max_images >= 0 and i >= max_images: break
      for pair in img['qa_pairs']:
        out = {}
        out['image'] = self._getImage(img)
        out['mc'] = self._getQAMultipleChoice(pair, shuffle)
        yield out

  def iterImageQAPairBatch(self, split = 'train', max_images = -1, max_batch_size = 100):
    batch = []
    for i, img in enumerate(self.split[split]):
      if max_images >= 0 and i >= max_images: break
      for pair in img['qa_pairs']:
        out = {}
        out['image'] = self._getImage(img)
        out['qa_pair'] = self._getQAPair(pair)
        batch.append(out)
        if len(batch) >= max_batch_size:
          yield batch
          batch = []
    if batch:
      yield batch

  def iterQAMultipleChoice(self, split = 'train', shuffle = False):
    for img in self.split[split]:
      for pair in img['qa_pairs']:
        yield self._getQAMultipleChoice(pair, shuffle)

  def iterQAPairs(self, split = 'train'):
    for img in self.split[split]:
      for pair in img['qa_pairs']:
        yield self._getQAPair(pair)

  def iterImages(self, split = 'train', shuffle = False, max_images = -1):
    imglist = self.split[split]
    ix = range(len(imglist))
    if shuffle:
      random.shuffle(ix)
    if max_images > 0:
      ix = ix[:min(len(ix), max_images)] # crop the list
    for i in ix:
      yield self._getImage(imglist[i])

def getDataProvider(dataset, **kwargs):
  """ we could intercept a special dataset and return different data providers """
  assert dataset in ['visual7w-telling'], 'dataset %s unknown' % (dataset, )
  return BasicDataProvider(dataset, **kwargs)
