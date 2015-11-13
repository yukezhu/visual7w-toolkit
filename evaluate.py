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
      logging.error('top_k is set to 1 for multiple-choice QA')
      top_k = 1
  else:
    logging.info('Open-ended QA evaluation')

  # split to be evaluated
  split = params['split']
  if split == 'test':
    logging.error('Please use our online server for test set evaluation.')
    return
  
  if split not in ['train', 'val']:
    logging.error('Error: cannot find split %s.' % split)
    return

  # load result json
  result_file = params['results']
  if os.path.isfile(result_file):
    results = json.load(open(result_file))
  else:
    logging.error('Error: cannot read result file from %s' % result_file)
    return
  
  # initialize counters
  num_correct = 0
  num_total = 0
  
  # fetch all test QA pairs from data provider
  pairs = {pair['qa_id']: pair for pair in dp.iterQAPairs(split)}
  
  # question_categories
  question_categories = ['what', 'where', 'when', 'who', 'why', 'how']
  category_total = dict()
  category_correct = dict()
  
  # loop through each prediction and check with ground-truth
  for idx, entry in enumerate(results):
    if entry['qa_id'] not in pairs:
      logging.error('Cannot find QA #%d. Are you using the correct split?' % entry['qa_id'])
      return
    pair = pairs[entry['qa_id']]
    answer_tokens = pair['answer_tokens']
    candidates = entry['candidates'][:top_k]
    correct_prediction = False
    for candidate in candidates:
      prediction = candidate['answer']
      if not prediction.endswith('.'): prediction += '.'
      prediction_tokens = dp.tokenize(prediction, 'answer')
      if prediction_tokens == answer_tokens:
        num_correct += 1
        correct_prediction = True
        break
    for c in question_categories:
      if pair['question'].lower().startswith(c):
        category_total[c] = category_total.get(c, 0) + 1
        if correct_prediction: category_correct[c] = category_correct.get(c, 0) + 1
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
    for c in question_categories:
      total = category_total.get(c, 0)
      correct = category_correct.get(c, 0)
      logging.info('Question type "%s" accuracy = %.3f (%d / %d)' % (c, 1.0 * correct / total, correct, total))
  
if __name__ == '__main__':
  
  # configure logging settings
  FORMAT = "%(asctime)-15s %(message)s"
  logging.basicConfig(format=FORMAT, level=logging.DEBUG)
  
  # configure argument parser
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--dataset', default='visual6w', type=str, help='dataset name (default: visual6w)')
  parser.add_argument('-m', '--mode', default='open', type=str, help='prediction mode. "mc" denotes multiple-choice QA. "open" denotes open-ended QA.')
  parser.add_argument('-k', '--topk', default=1, type=int, help='top k evaluation. k denotes how many candidate answers to be examined.')
  parser.add_argument('-j', '--results', default='results/result_visual6w_open.json', help='path to json file contains the results (see the format of the sample files in "results" folder).')
  parser.add_argument('-o', '--output_path', default='.', type=str, help='output folder')
  parser.add_argument('-s', '--split', type=str, default='val', help='the split to be evaluated: train / val / test (default: val)')
  parser.add_argument('-v', '--verbose', default=0, type=int, help='verbose mode. print performances of 6W categories when enabled.')

  # parse arguments
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  # load dataset (skipping feature files)
  dp = getDataProvider(params['dataset'], load_features = False)
  
  # start evaluation mode
  if params['mode'] in ['mc', 'open']:
    evaluate_top_k(dp, params)
  else:
    logging.error('Error: evaluation mode "%s" is not supported.' % params['mode'])
