import json
import os
import random
import scipy.io
from collections import defaultdict

class BasicDataProvider:
  def __init__(self, dataset, **kwargs):
    print 'Initializing data provider for dataset %s...' % (dataset, )

    # !assumptions on folder structure
    self.dataset_root = kwargs.get('dataset_root', os.path.join('datasets', dataset))
    self.feature_root = kwargs.get('feature_root', os.path.join('datasets', dataset))
    self.image_root = kwargs.get('image_root', os.path.join('data', 'images', dataset))

    # load the dataset into memory
    dataset_path = os.path.join(self.dataset_root, 'dataset.json')
    print 'BasicDataProvider: reading %s' % (dataset_path, )
    self.dataset = json.load(open(dataset_path, 'r'))

    # load the image features into memory
    self.load_features = kwargs.get('load_features', True)
    if self.load_features:
      # load feature
      features_path = os.path.join(self.feature_root, 'vgg_fc7_feats.mat')
      print 'BasicDataProvider: reading %s' % (features_path, )
      features_struct = scipy.io.loadmat(open(features_path, 'rb'))
      self.features = features_struct['feats']
      # imgid2featidx is a dictionary that maps an image id to the column index of the feature matrix
      image_ids = features_struct['image_ids'].ravel()
      self.imgid2featidx = {img : i for i, img in enumerate(image_ids)}

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
    # lazily fill in some attributes
    if self.load_features and not 'feat' in img: # also fill in the features
      feature_index = self.imgid2featidx[img['image_id']]
      img['feat'] = self.features[:,feature_index]
    return img

  def _getQAPair(self, qa_pair):
    """ create a QA pair structure """
    if not 'tokens' in qa_pair:
      question_tokens = self.tokenize(qa_pair['question'], 'question')
      qa_pair['question_tokens'] = question_tokens
      if 'answer' in qa_pair:
        answer_tokens = self.tokenize(qa_pair['answer'], 'answer')
        qa_pair['answer_tokens'] = answer_tokens
        qa_pair['tokens'] = question_tokens + answer_tokens
    return qa_pair
  
  def _getQAMultipleChoice(self, qa_pair, shuffle = False):
    """ create a QA multiple choice structure """
    qa_pair = self._getQAPair(qa_pair)
    if 'multiple_choices' in qa_pair:
      mcs = qa_pair['multiple_choices']
      tokens = [self.tokenize(x, 'answer') for x in mcs]
      pos_idx = range(len(mcs)+1)
      # random shuffle the positions of multiple choices
      if shuffle:
        random.shuffle(pos_idx)
      qa_pair['mc_tokens'] = []
      qa_pair['mc_candidates'] = []
      for idx, k in enumerate(pos_idx):
        if k == 0 and 'answer' in qa_pair:
          qa_pair['mc_tokens'].append(qa_pair['answer_tokens'])
          qa_pair['mc_candidates'].append(qa_pair['answer'])
          qa_pair['mc_selection'] = idx # record the position of the true answer
        else:
          qa_pair['mc_tokens'].append(tokens[k-1])
          qa_pair['mc_candidates'].append(mcs[k-1])
    return qa_pair

  # PUBLIC FUNCTIONS
  def tokenize(self, sent, token_type=None):
    """ convert question or answer into a sequence of tokens """
    line = sent[:-1].lower().replace('.', '')
    line = ''.join([x if x.isalnum() else ' ' for x in line])
    tokens = line.strip().split()
    if token_type == 'question':
      assert sent[-1] == '?', 'question (%s) must end with question mark.' % sent
      tokens.append('?')
    if token_type == 'answer':
      assert sent[-1] == '.', 'answer (%s) must end with period.' % sent
      tokens.append('.')
    return tokens
  
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
    """ sample image QA pair from a split """
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
  assert dataset in ['visual6w'], 'dataset %s unknown' % (dataset, )
  return BasicDataProvider(dataset, **kwargs)
