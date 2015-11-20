import os
import argparse
import logging
import json

from common.data_provider import getDataProvider

"""
  Compare top K candidate predictions with ground-truth answers
  We say that a model predicts the correct answer if one of the
  top k predictions match exactly with the ground-truth answers.
  Accuracy is used to report the performance.

  When evaluating multiple-choice QAs, the model makes a single
  prediction (i.e., the multiple-choice option it selects).

  - dp: data provider (an access helper to QA dataset)
  - params: evaluation mode configurations
"""
def evaluate_top_k(dp, params):
  # set parameter
  top_k = params['topk']
  if params['mode'] == 'mc':
    logging.info('Multiple-choice QA evaluation')
    if top_k != 1:
      logging.info('top_k is set to 1 for multiple-choice QA')
      top_k = 1
  else:
    logging.info('Open-ended QA evaluation')

  # split to be evaluated
  split = params['split']
  if split not in ['train', 'val', 'test']:
    logging.error('Error: cannot find split %s.' % split)
    return

  # load result json
  result_file = params['results']
  try:
    results = json.load(open(result_file))
  except:
    logging.error('Error: cannot read result file from %s' % result_file)
    return
  
  # initialize counters
  num_correct = 0
  num_total = 0
  
  # fetch all test QA pairs from data provider
  pairs = {pair['qa_id']: pair for pair in dp.iterQAPairs(split)}
  
  # record performances per question category
  category_total = dict()
  category_correct = dict()
  
  # loop through each prediction and check with ground-truth
  for idx, entry in enumerate(results):
    if entry['qa_id'] not in pairs:
      logging.error('Cannot find QA #%d. Are you using the correct split?' % entry['qa_id'])
      return
    pair = pairs[entry['qa_id']]
    answer = str(pair['answer']).lower()
    candidates = entry['candidates'][:top_k]
    c = pair['type']
    category_total[c] = category_total.get(c, 0) + 1
    for candidate in candidates:
      prediction = str(candidate['answer']).lower()
      if prediction == answer:
        num_correct += 1
        category_correct[c] = category_correct.get(c, 0) + 1
        break
    num_total += 1
    if (idx+1) % 10000 == 0:
      logging.info('Evaluated %s QA pairs...' % format(idx+1, ',d'))
  
  # compute metrics
  accuracy = 1.0 * num_correct / num_total
  logging.info('Done!\n')
  logging.info('Evaluated on %s QA pairs with top-%d predictions.' % (format(num_total, ',d'), top_k))
  logging.info('Overall accuracy = %.3f' % accuracy)
  
  verbose = params['verbose']
  if verbose:
    for c in category_total.keys():
      total = category_total.get(c, 0)
      correct = category_correct.get(c, 0)
      logging.info('Question type "%s" accuracy = %.3f (%d / %d)' % (c, 1.0 * correct / total, correct, total))
  
if __name__ == '__main__':
  
  # configure logging settings
  FORMAT = "%(asctime)-15s %(message)s"
  logging.basicConfig(format=FORMAT, level=logging.DEBUG)
  
  # configure argument parser
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--dataset', default='visual7w-telling', type=str, help='dataset name (default: visual7w-telling)')
  parser.add_argument('-m', '--mode', default='open', type=str, help='prediction mode: "mc" - multiple-choice QA; "open" - open-ended QA.')
  parser.add_argument('-k', '--topk', default=1, type=int, help='top-k evaluation. k is the number of answer candidates to be examined.')
  parser.add_argument('-j', '--results', default='results/result_visual7w-telling_open.json', help='path to json file contains the results')
  parser.add_argument('-s', '--split', type=str, default='val', help='the split to be evaluated: train / val / test (default: val)')
  parser.add_argument('-v', '--verbose', default=0, type=int, help='verbose mode. report performances of question categories when enabled.')

  # parse arguments
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  # load dataset (skipping feature files)
  dp = getDataProvider(params['dataset'])
  
  # start evaluation mode
  if params['dataset'].endswith('telling'):
    # multiple-choice and open-ended evaluations are supported in telling QA
    assert params['mode'] in ['mc', 'open'], 'Evaluation mode %s not supported in telling QA.' % params['mode']
    evaluate_top_k(dp, params)
  elif params['dataset'].endswith('pointing'):
    # only multiple-choice evaluation is supported in pointing QA
    assert params['mode'] in ['mc'], 'Evaluation mode %s not supported in pointing QA.' % params['mode']
    evaluate_top_k(dp, params)
  else:
    logging.error('Error: evaluation mode "%s" is not supported.' % params['mode'])
