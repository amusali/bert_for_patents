## initialize model

## Load BERT
from bert import tokenization
from api.bertfns import BertPredictor
import tensorflow as tf
import time
from absl import flags
import re
import os
from path_utils import get_base_path

base_path = get_base_path()

flags.FLAGS([""])

## Initialzie BERT
MAX_SEQ_LENGTH = 512
MAX_PREDS_PER_SEQUENCE = 45
if 'COLAB_GPU' in os.environ:
  MODEL_DIR = "/content/drive/MyDrive/bert_large_trained_on_patents/temp_dir/rawout"
else:
  MODEL_DIR = os.path.join(base_path, "bert_large_trained_on_patents/temp_dir/rawout")
VOCAB = os.path.join(base_path, "bert_for_patents_vocab_39k.txt")
return_cls_embedding = True

pooling = tf.keras.layers.GlobalAveragePooling1D()

tokenizer = tokenization.FullTokenizer(VOCAB, do_lower_case=True)
bert_predictor = BertPredictor(
    model_name=MODEL_DIR,
    text_tokenizer=tokenizer,
    max_seq_length=MAX_SEQ_LENGTH,
    max_preds_per_seq=MAX_PREDS_PER_SEQUENCE,
    has_context=True)

print('BERT loaded...')

def get_embd_of_whole_abstract(abstracts, return_cls_embedding=True, has_context_token=True):
    """
    Function to handle both single abstracts and batches of abstracts.
    
    abstracts: Can be a single string (abstract) or a list of strings (multiple abstracts).
    return_cls_embedding: If True, returns the CLS token embedding.
    has_context_token: If True, adds 'abstract' as a context token.
    """
    
    # If a single abstract (string) is passed, convert it to a list of one element
    if isinstance(abstracts, str):
        abstracts = [abstracts]
    
    start_time = time.time()
    
    ## Add context tokens if needed
    if has_context_token:
        context_tokens = ['abstract']
    else:
        context_tokens = []
    
    ## Predict -- get embeddings for the batch of abstracts
    response = bert_predictor.predict(
        abstracts, context_tokens=context_tokens * len(abstracts))

    # Collect the CLS embeddings for each abstract in the batch
    cls_token_embeddings = response[0]['cls_token'].numpy()  # Shape will be (batch_size, 1024)

    end_time = time.time()
    average_time = (end_time - start_time)/len(abstracts)
    print(f'There are {len(abstracts)} patents being fed into BERT')

    print(f"It took {average_time:.5f} seconds to get the embeddings of the input.")
    
    if return_cls_embedding:
        # Return CLS embeddings for all abstracts in the batch
        return cls_token_embeddings  # Directly return the array of embeddings (shape: batch_size x 1024)
    else:
        # If not using CLS, perform pooling over the entire embedding (encoder_layer)
        encoder_layers = response[0]['encoder_layer'].numpy()  # Shape: (batch_size, seq_length, 1024)
        avg_embeddings = pooling(tf.reshape(encoder_layers, shape=[encoder_layers.shape[0], -1, 1024]))
        return avg_embeddings.numpy()


  
# Function to tokenize abstract into a list of sentences using regex
def tokenize_abstract_with_regex(abstract):
    # Regular expression to match sentence boundaries
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', abstract)
    return sentences


