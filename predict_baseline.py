import argparse
import operator
import json
import random
import os

from common.data_provider import getDataProvider

def main_freefrom_most_frequent_answers(params):
  """ 
    Open-ended QA baseline: 
      take the top five most frequent answers in the trianing set
      as the predicted answers to all the questions
  """
  # load the checkpoint
  result_path = params['result_path']
  dataset = params['dataset']
  topk = params['topk']
  split = params['split']
  
  # fetch the data provider
  dp = getDataProvider(dataset)

  # initialize
  blob = []

  # count answer frequencies
  answer_freqs = dict()
  for pair in dp.iterImageQAPair(split='train'):
    answer = pair['qa_pair']['answer']
    answer_freqs[answer] = answer_freqs.get(answer, 0) + 1
  
  sorted_freqs = sorted(answer_freqs.items(), key=operator.itemgetter(1), reverse=True)
  top_answers = [x[0] for x in sorted_freqs[:topk]]
  
  # iterate over all QAs and predict the answers
  for pair in dp.iterImageQAPair(split=split):

    # build up the output
    img_blob = {}
    img_blob['question'] = pair['qa_pair']['question']
    img_blob['qa_id'] = pair['qa_pair']['qa_id']
    img_blob['candidates'] = []

    # add the frequent answers as prediction
    for answer in top_answers:
      img_blob['candidates'].append({'answer': answer})

    blob.append(img_blob)

  # dump result struct to file
  save_file = os.path.join(result_path, 'result_%s_open.json' % dataset)
  print 'writing predictions to %s...' % (save_file, )
  json.dump(blob, open(save_file, 'w'), indent=2)

def main_multiple_choice_random_guess(params):
  """
    Multiple-choice QA baseline:
      uniformly sample an answer from the pool of multiple choices
  """
  # load the checkpoint
  result_path = params['result_path']
  dataset = params['dataset']
  split = params['split']
  
  # fetch the data provider
  dp = getDataProvider(dataset)
  
  # initialize
  blob = []
  
  # iterate over all QAs and predict the answers
  for mc in dp.iterImageQAMultipleChoice(split=split, shuffle=True):
    
    # build up the output
    img_blob = {}
    img_blob['question'] = mc['mc']['question']
    img_blob['qa_id'] = mc['mc']['qa_id']
    img_blob['candidates'] = []

    # make a random choice
    prediction = random.choice(mc['mc']['mc_candidates'])
    img_blob['candidates'].append({'answer': prediction})

    blob.append(img_blob)
  
  # dump result struct to file
  save_file = os.path.join(result_path, 'result_%s_mc.json' % dataset)
  print 'writing predictions to %s...' % (save_file, )
  json.dump(blob, open(save_file, 'w'), indent=2)

def main_multiple_choice_most_frequent(params):
  """
    Multiple-choice QA baseline:
      select the most frequent answer from the multiple choices
      as the prediction (the frequencies are counted in the training set)
  """
  # load the checkpoint
  result_path = params['result_path']
  dataset = params['dataset']
  split = params['split']
  
  # fetch the data provider
  dp = getDataProvider(dataset)
  
  # count answer frequencies
  answer_freqs = dict()
  for pair in dp.iterImageQAPair(split='train'):
    answer = pair['qa_pair']['answer']
    answer_freqs[answer] = answer_freqs.get(answer, 0) + 1
  
  # initialize
  blob = []
  
  # iterate over all QAs and predict the answers
  for mc in dp.iterImageQAMultipleChoice(split=split, shuffle=True):
    
    # build up the output
    img_blob = {}
    img_blob['question'] = mc['mc']['question']
    img_blob['qa_id'] = mc['mc']['qa_id']
    img_blob['candidates'] = []

    # make the prediction as the most frequent answer
    max_freq = -1
    max_k = -1
    for k, mc in enumerate(mc['mc']['mc_candidates']):
      freq = answer_freqs.get(mc, 0)
      if freq > max_freq:
        max_freq = freq
        max_k = k
        prediction = mc
    
    img_blob['candidates'].append({'answer': prediction})

    blob.append(img_blob)
  
  # dump result struct to file
  save_file = os.path.join(result_path, 'result_%s_mc.json' % dataset)
  print 'writing predictions to %s...' % (save_file, )
  json.dump(blob, open(save_file, 'w'), indent=2)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--dataset', type=str, default='visual7w-telling', help='dataset name (default: visual7w-telling).')
  parser.add_argument('-r', '--result_path', default='results', type=str, help='folder to store prediction results (default: results)')
  parser.add_argument('-m', '--mode', type=str, default='open', help='prediction mode: open / mc (default: open)')
  parser.add_argument('-k', '--topk', type=int, default=5, help='only used for open-ended evaluation. use the top k most frequent answers as the predictions (default: 5)')
  parser.add_argument('-s', '--split', type=str, default='val', help='the split to be evaluated: train / val / test (default: val)')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  
  print 'parsed parameters:'
  print json.dumps(params, indent = 2)
  
  # start evaluation mode
  if params['dataset'].endswith('telling'):
    # multiple-choice and open-ended evaluations are supported in telling QA
    assert params['mode'] in ['mc', 'open'], 'Evaluation mode %s not supported in telling QA.' % params['mode']
    if params['mode'] == 'mc':
      main_multiple_choice_most_frequent(params)
    elif params['mode'] == 'open':
      main_freefrom_most_frequent_answers(params)
  elif params['dataset'].endswith('pointing'):
    # only multiple-choice evaluation is supported in pointing QA
    assert params['mode'] in ['mc'], 'Evaluation mode %s not supported in pointing QA.' % params['mode']
    main_multiple_choice_random_guess(params)
  else:
    print 'Error: unsupported evaluation mode "%s"' % params['mode']
