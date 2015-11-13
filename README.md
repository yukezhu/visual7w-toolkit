# Visual7W Toolkit

![alt text](http://web.stanford.edu/~yukez/images/img/visual7w_examples.png "Visual7W example QAs")

## Introduction

[Visual7W](http://web.stanford.edu/~yukez/visual7w.html) is a large-scale visual question answering (QA) dataset, with object-level groundings and multimodal answers.
Each question starts with one of the seven Ws, *what*, *where*, *when*, *who*, *why*, *how* and *which*.
Please check out [our arxiv paper](http://web.stanford.edu/~yukez/papers/visual7w_arxiv.pdf) for more details.
This toolkit is used for parsing dataset files and evaluating model performances.
Please contact [Yuke Zhu](http://web.stanford.edu/~yukez/) for questions, comments, or bug reports.

## Dataset Overview

The Visual7W dataset is collected on 47,300 COCO images. In total, it has 327,939 QA pairs, together with 1,311,756 human-generated multiple-choices and 561,459 object groundings from 36,579 categories. In addition, we provide complete grounding annotations that link the object mentions in the QA sentences to their bounding boxes in the images and therefore
introduce a new QA type with image regions as the visually grounded answers. We refer to questions with textual answers
as *telling* QA and to such with visual answers as *pointing* QA. The figure above shows some examples in the Visual7W dataset, where the first row shows *telling* QA examples, and the second row shows *pointing* QA examples.

## Evaluation Methods

We use two evaluation methods to measure performance. **Multiple-choice evaluation** aims at selecting the correct option from a pre-defined pool of candidate answers. **Open-ended evaluation** aims at predicting a freeform textual answer given a question and the image. This toolkit provides utility functions to evaluate performances in both methods. We explain the details of these two methods below.

1. **Multiple-choice QA**: We provide four human-generated multiple-choice answers for each question, where one of them is the ground-truth. We say the model is correct on a question if it selects the correct answer candidate. Accuracy is used to measure the performance. This is the default (and recommended) evaluation method for Visual7W.

2. **Open-ended QA**: similar to the top-5 criteria used in [ImageNet challenges](http://www.image-net.org/), we let the model to make *k* different freeform predictions. We say the model is correct on a question if one of the *k* predictions matches exactly with the ground-truth. Accuracy is used to measure the performance. This evlaution method only applies to the *telling* QA tasks with textual answers.

## How to Use

Before using this toolkit, make sure that you have downloaded the Visual7W dataset. 
You can use our downloading script in ```datasets/[dataset-name]/download_dataset.sh``` 
to fetch the database json to the local disk.

### Telling QA

We implement a most-frequent-answer (MFA) baseline in ```predict_baseline.py```.
For open-ended evaluation, we use the top-*k* most frequent training set answers 
as the predictions for all test questions. For multiple-choice evaluation, we select 
the candidate answer with the highest training set frequency for each test question.

In this demo, we perform open-ended evaluation for *telling* QA.
To run the MFA baseline on the validation set, use the following command:

```
python predict_baseline.py --dataset visual7w-telling \
                           --mode open \
                           --topk 100 \
                           --split val \
                           --result_path results
```

It will generate a prediction file ```result_visual7w-telling_open.json``` in the ```results``` folder. Type ```python predict_baseline.py -h``` to learn more about the input arguments.

The script below shows how to use the evaluation script ```evaluate.py``` to check the performances of the open-ended predictions in the ```result_visual7w-telling_open.json``` file. Type ```python evaluate.py -h``` to learn more about the input arguments.

```
python evaluate.py --dataset visual7w-telling \
                   --mode open \
                   --topk 100 \
                   --split val \
                   --results results/result_visual7w-telling_open.json \
                   --verbose 1
```

You will see the similar results as below:

```
2015-11-13 14:42:42,259 Evaluated on 28,020 QA pairs with top-100 predictions.
2015-11-13 14:42:42,259 Overall accuracy = 0.371
2015-11-13 14:42:42,259 Question type "what" accuracy = 0.380 (5057 / 13296)
2015-11-13 14:42:42,259 Question type "who" accuracy = 0.378 (1087 / 2879)
2015-11-13 14:42:42,259 Question type "when" accuracy = 0.529 (668 / 1262)
2015-11-13 14:42:42,259 Question type "how" accuracy = 0.721 (3037 / 4211)
2015-11-13 14:42:42,259 Question type "where" accuracy = 0.100 (459 / 4590)
2015-11-13 14:42:42,259 Question type "why" accuracy = 0.051 (91 / 1782)
```

Change the ```mode``` parameter to ```mc``` when performing multiple-choice evaluation.

### Pointing QA

Similary we can use the toolkit to evaluate pointing QA. For demo purpose, we implement a very simple baseline, which picks a random answer out of the four multiple-choice candidates.
You can run the baseline as follows. Please make sure that you have downloaded the dataset json before running the code.

```
python predict_baseline.py --dataset visual7w-pointing \
                           --mode mc \
                           --split val \
                           --result_path results
```

You should expect something very close to chance performance (25%). Let's see if that is true.

```
python evaluate.py --dataset visual7w-pointing \
                   --mode mc \
                   --split val \
                   --results results/result_visual7w-pointing_mc.json \
                   --verbose 1
```

Here is what I got.

```
2015-11-13 14:45:56,363 Evaluated on 36,990 QA pairs with top-1 predictions.
2015-11-13 14:45:56,363 Overall accuracy = 0.249
2015-11-13 14:45:56,363 Question type "which" accuracy = 0.249 (9209 / 36990)
```

### Evaluating Your Own Models

In order to evaluate your own model, please check the format of the sample outputs 
produced by the baseline script.  In short,
a prediction file contains a list of predicted answers in the ```candidates``` arrays. 
For multiple-choice QA, the ```candidates``` arrays contain only one element, which is 
the selected multiple-choice option. For open-ended QA, the ```candidates``` arrays can 
contain more than one (up to *k*) predictions, where we use the one-of-*k* metric to 
evaluate the performance.
