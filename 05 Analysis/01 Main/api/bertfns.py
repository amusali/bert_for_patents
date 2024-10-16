import collections
import math
import random
import sys
import time
from typing import Dict, List, Tuple

# Use Tensorflow 2.0
import tensorflow as tf
import numpy as np
#import tokenization     
from bert import tokenization


# The functions in this block are also found in the bert cloned repo in the 
# `run_classifier.py` file, however those also have some compatibility issues 
# and thus the functions needed are just copied here.

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_ids, input_mask, segment_ids, label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample."""
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)
    features.append(feature)
  return features

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)
  #print(tokens)
  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature


#@hidden_cell
def get_tokenized_input(
    texts: List[str], tokenizer: tokenization.FullTokenizer) -> List[List[int]]:
  """Returns list of tokenized text segments."""

  return [tokenizer.tokenize(text) for text in texts]

## BERT Predictor 
class BertPredictor():

  def __init__(
      self, 
      model_name: str, 
      text_tokenizer: tokenization.FullTokenizer, 
      max_seq_length: int,
      max_preds_per_seq: int,
      has_context: bool = False):
    """Initializes a BertPredictor object."""

    self.tokenizer = text_tokenizer
    self.max_seq_length = max_seq_length
    self.max_preds_per_seq = max_preds_per_seq
    self.mask_token_id = 4
    # If you want to add context tokens to the input, set value to True.
    self.context = has_context

    model = tf.compat.v2.saved_model.load(export_dir=model_name, tags=['serve'])
    self.model = model.signatures['serving_default']

  def get_features_from_texts(self, texts: List[str]) -> Dict[str, int]:
    """Uses tokenizer to convert raw text into features for prediction."""

    #examples = [run_classifier.InputExample(0, t, label='') for t in texts]
    #features = run_classifier.convert_examples_to_features(
    #    examples, [''], self.max_seq_length, self.tokenizer)
    examples = [InputExample(0, t, label='') for t in texts]
    features = convert_examples_to_features(
        examples, [''], self.max_seq_length, self.tokenizer)
    return dict(
        input_ids=[f.input_ids for f in features],
        input_mask=[f.input_mask for f in features],
        segment_ids=[f.segment_ids for f in features]
    )

  def insert_token(self, input: List[int], token: int) -> List[int]:
    """Adds token to input."""

    return input[:1] + [token] + input[1:-1]

  def add_input_context(
      self, inputs: Dict[str, int], context_tokens: List[str]
  ) -> Dict[str, int]:
    """Adds context token to input features."""

    context_token_ids = self.tokenizer.convert_tokens_to_ids(context_tokens)
    segment_token_id = 0
    mask_token_id = 1

    for i, context_token_id in enumerate(context_token_ids):
      inputs['input_ids'][i] = self.insert_token(
          inputs['input_ids'][i], context_token_id)

      inputs['segment_ids'][i] = self.insert_token(
          inputs['segment_ids'][i], segment_token_id)

      inputs['input_mask'][i] = self.insert_token(
          inputs['input_mask'][i], mask_token_id)
    return inputs

  def create_mlm_mask(
      self, inputs: Dict[str, int], mlm_ids: List[List[int]]
  ) -> Tuple[Dict[str, List[List[int]]], List[List[str]]]:
    """Creates masked language model mask."""

    masked_text_tokens = []
    mlm_positions = []

    if not mlm_ids:
      inputs['mlm_ids'] = mlm_positions
      return inputs, masked_text_tokens

    for i, _ in enumerate(mlm_ids):

      masked_text = []

      # Pad mlm positions to max seqeuence length.
      mlm_positions.append(
          mlm_ids[i] + [0] * (self.max_preds_per_seq - len(mlm_ids[i])))

      for pos in mlm_ids[i]:
        # Retrieve the masked token.
        masked_text.extend(
            self.tokenizer.convert_ids_to_tokens([inputs['input_ids'][i][pos]]))
        # Replace the mask positions with the mask token.
        inputs['input_ids'][i][pos] = self.mask_token_id
  
      masked_text_tokens.append(masked_text)

    inputs['mlm_ids'] = mlm_positions
    return inputs, masked_text_tokens

  def predict(
      self, texts: List[str], mlm_ids: List[List[int]] = None, 
      context_tokens: List[str] = None
  ) -> Tuple[Dict[str, tf.Tensor], Dict[str, List[List[int]]], List[List[str]]]:
    """Gets BERT predictions for provided text and masks.
    
    Args:
      texts: List of texts to get BERT predictions.
      mlm_ids: List of lists corresponding to the mask positions for each input
        in `texts`.
      context_token: List of string contexts to prepend to input texts.

    Returns:
      response: BERT model response.
      inputs: Tokenized and modified input to BERT model.
      masked_text: Raw strings of the masked tokens.
    """

    if mlm_ids:
      assert len(mlm_ids) == len(texts), ('If mask ids provided, they must be '
          'equal to the length of the input text.')

    if self.context:
      # If model uses context, but none provided, use 'UNK' token for context.
      if not context_tokens:
        context_tokens = ['[UNK]' for _ in range(len(texts))]
      #print("Length of text: ", len(texts))
      #print("Length of context tokens: ", len(context_tokens))
      assert len(context_tokens) == len(texts), ('If context tokens provided, '
          'they must be equal to the length of the input text.')
    
    inputs = self.get_features_from_texts(texts)

    # If using a BERT model with context, add corresponding tokens.
    if self.context:
      inputs = self.add_input_context(inputs, context_tokens)

    inputs, masked_text = self.create_mlm_mask(inputs, mlm_ids)

    response = self.model(
      segment_ids=tf.convert_to_tensor(inputs['segment_ids'], dtype=tf.int64),
      input_mask=tf.convert_to_tensor(inputs['input_mask'], dtype=tf.int64),
      input_ids=tf.convert_to_tensor(inputs['input_ids'], dtype=tf.int64),
      mlm_positions=tf.convert_to_tensor(inputs['mlm_ids'], dtype=tf.int64),
      )
    
    if mlm_ids:
      # Do a reshape of the mlm logits (batch size, num predictions, vocab).
      new_shape = (len(texts), self.max_preds_per_seq, -1)
      response['mlm_logits'] = tf.reshape(
          response['mlm_logits'], shape=new_shape)
    
    return response, inputs, masked_text 
  

# Some helper functions.

def get_mlm_ids_by_token(
    mask_token: str, tokenized_text: List[List[str]], 
    has_context: bool = False, first_occurence: bool = True
) -> List[List[int]]:
  """Returns position ids for masking a specified token."""

  pos_add = 2 if has_context else 1
  mlm_ids = []
  for i, tokens in enumerate(tokenized_text):
    pub_mlm_ids = []
    for j, token in enumerate(tokens):
      if token == mask_token:
        pub_mlm_ids.append(j + pos_add)
        if first_occurence:
          break
    mlm_ids.append(pub_mlm_ids)

  return mlm_ids


def bert_topk_predictions(
    mlm_logits: tf.Tensor, mlm_ids: List[List[int]], top_k: int = 5
) -> Tuple[List[int], List[str]]:
  """Returns BERT predicted token ids and terms for masked ids.
  
  Args:
    mlm_logits: The BERT masked language logits.
    mlm_ids: The masked ids.
    top_k: Number of predictions to return for each mask.

  Returns:
    token_preds: Token predictions for each mask position.
    term_preds: Term predictions for each mask position.
  """

  token_preds = []
  term_preds = []

  # Tradeoff between single call for all (including non masked) and then gather
  # vs. calling math top_k over and over

  for i, ids in enumerate(mlm_ids):
    current_token_preds = []
    current_term_preds = []
    for j, id in enumerate(ids):
      preds = tf.math.top_k(mlm_logits[i][j], top_k).indices.numpy().tolist()
      current_token_preds.append(preds)
      current_term_preds.append(tokenizer.convert_ids_to_tokens(preds))
    token_preds.append(current_token_preds)
    term_preds.append(current_term_preds)

  return token_preds, term_preds


def find_rankings(
    words: List[str], word_ids: List[int], mlm_logits: tf.Tensor, 
    mlm_ids: List[List[str]]
) -> Dict[str, float]:
  """Return the rankings in the bert predictions for the provided words."""
  
  word_positions = []

  # Iterate through all predictions.
  for i, _ in enumerate(mlm_ids):
    for j, _ in enumerate(mlm_ids[i]):
      logits = tf.argsort(mlm_logits[i][j], direction='DESCENDING')
      positions = tf.reshape(tf.where(tf.equal(
          tf.expand_dims(word_ids, axis=-1), logits))[:,-1], [1, -1])
      word_positions.extend(list(positions.numpy()))

  transposed = np.array(word_positions).T
  word_dict = dict()

  for i, word in enumerate(words):
    total = sum(transposed[i])
    word_dict[word] = {
        'average': transposed[i].mean(),
        'max': transposed[i].max(),
        'min': transposed[i].min(),
        'std': transposed[i].std(),
    }

  return word_dict
