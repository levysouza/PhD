import warnings
import io
import random
import numpy as np
import mxnet as mx
import gluonnlp as nlp
from bert import data, model
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from heapq import nsmallest

#warnings.filterwarnings('ignore')

print('ok1')

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)
# change `ctx` to `mx.cpu()` if no GPU is available.
ctx = mx.cpu()

# read_file = pd.read_csv('dataset/train_dataset_1_1', delimiter=',', header=None)
# train_dataset = read_file.iloc[:,:].values

print('ok2')

bert_base, vocabulary = nlp.model.get_model('bert_12_768_12',
                                             dataset_name='book_corpus_wiki_en_uncased',
                                             pretrained=True, ctx=ctx, use_pooler=True,
                                             use_decoder=False, use_classifier=False)

print('ok3')

# bert_classifier = model.classification.BERTClassifier(bert_base, num_classes=2, dropout=0.1)
# # only need to initialize the classifier layer.
# bert_classifier.classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
# bert_classifier.hybridize(static_alloc=True)

# # softmax cross entropy loss for classification
# loss_function = mx.gluon.loss.SoftmaxCELoss()
# loss_function.hybridize(static_alloc=True)

# metric = mx.metric.Accuracy()

# # Skip the first line, which is the schema
# num_discard_samples = 1
# # Split fields by tabs
# field_separator = nlp.data.Splitter('\t')
# # Fields to select from the file
# field_indices = [0, 1, 2]
# data_train_raw = nlp.data.TSVDataset(filename='fine_tuning_data.tsv',
#                                  field_separator=field_separator,
#                                  num_discard_samples=num_discard_samples,
#                                  field_indices=field_indices)

print('ok4')
print('ok2')
# Use the vocabulary from pre-trained model for tokenization
# bert_tokenizer = nlp.data.BERTTokenizer(vocabulary, lower=True)

# # The maximum length of an input sequence
# max_len = 128

# # The labels for the two classes [(0 = not similar) or  (1 = similar)]
# all_labels = ["0", "1"]

# # whether to transform the data as sentence pairs.
# # for single sentence classification, set pair=False
# # for regression task, set class_labels=None
# # for inference without label available, set has_label=False
# pair = True
# transform = data.transform.BERTDatasetTransform(bert_tokenizer, max_len,
#                                                 class_labels=all_labels,
#                                                 has_label=True,
#                                                 pad=True,
#                                                 pair=pair)
# data_train = data_train_raw.transform(transform)


print('ok5')

# batch_size = 32
# lr = 5e-6

# # The FixedBucketSampler and the DataLoader for making the mini-batches
# train_sampler = nlp.data.FixedBucketSampler(lengths=[int(item[1]) for item in data_train],
#                                             batch_size=batch_size,
#                                             shuffle=True)
# bert_dataloader = mx.gluon.data.DataLoader(data_train, batch_sampler=train_sampler)

# trainer = mx.gluon.Trainer(bert_classifier.collect_params(), 'adam',
#                            {'learning_rate': lr, 'epsilon': 1e-9})

# # Collect all differentiable parameters
# # `grad_req == 'null'` indicates no gradients are calculated (e.g. constant parameters)
# # The gradients for these params are clipped later
# params = [p for p in bert_classifier.collect_params().values() if p.grad_req != 'null']
# grad_clip = 1

# # Training the model with only three epochs
# log_interval = 4
# num_epochs = 4
# for epoch_id in range(num_epochs):
#     metric.reset()
#     step_loss = 0
#     for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(bert_dataloader):
#         with mx.autograd.record():

#             # Load the data to the GPU
#             token_ids = token_ids.as_in_context(ctx)
#             valid_length = valid_length.as_in_context(ctx)
#             segment_ids = segment_ids.as_in_context(ctx)
#             label = label.as_in_context(ctx)

#             # Forward computation
#             out = bert_classifier(token_ids, segment_ids, valid_length.astype('float32'))
#             ls = loss_function(out, label).mean()

#         # And backwards computation
#         ls.backward()

#         # Gradient clipping
#         trainer.allreduce_grads()
#         nlp.utils.clip_grad_global_norm(params, 1)
#         trainer.update(1)

#         step_loss += ls.asscalar()
#         metric.update([label], [out])

#         # Printing vital information
#         if (batch_id + 1) % (log_interval) == 0:
#             print('[Epoch {} Batch {}/{}] loss={:.4f}, lr={:.7f}, acc={:.3f}'
#                          .format(epoch_id, batch_id + 1, len(bert_dataloader),
#                                  step_loss / log_interval,
#                                  trainer.learning_rate, metric.get()[1]))
#             step_loss = 0


articles = pd.read_csv('dataset/test_dataset', delimiter=',', header=None)
data_articles = articles.iloc[:,:].values

read = pd.read_csv('dataset/cleanDataTables', delimiter=',', header=None)
data_tables = read.iloc[:,:].values

tokenizer = nlp.data.BERTTokenizer(vocabulary, lower=True);
transform = nlp.data.BERTSentenceTransform(tokenizer, max_seq_length=512, pair=False, pad=False);

articles_title = []
articles_id = []
article_dense_vector = []

for article_id, title, text in tqdm(data_articles):
    
    articles_id.append(article_id)
    
    articles_title.append(title)

    sample = transform(title)
    words, valid_len, segments = mx.nd.array([sample[0]]), mx.nd.array([sample[1]]), mx.nd.array([sample[2]])
    seq_encoding, cls_encoding = bert_base(words, segments, valid_len)
    
    article_dense_vector.append(cls_encoding[0].asnumpy())

tables_title = []
tables_dense_vector = []

for current_table in tqdm(data_tables):
    
    table_title = str(current_table[1])
    
    sample = transform(table_title)
    words, valid_len, segments = mx.nd.array([sample[0]]), mx.nd.array([sample[1]]), mx.nd.array([sample[2]])
    seq_encoding, cls_encoding = bert_base(words, segments, valid_len)
    
    tables_dense_vector.append(cls_encoding[0].asnumpy())

def get_id_ranked_tables(top_k,distance_vector):

    id_ranked_tables = []

    for current_top_k in top_k:
        
        index = np.where(distance_vector == current_top_k)
         
        index_colummun = index[0][0]
        
        id_ranked_tables.append(data_tables[index_colummun][0])

    return id_ranked_tables

def get_accuracy(id_ranked_tables, id_query_goal):

    accuracy = 0

    for id_table in id_ranked_tables:
    
        if id_table == id_query_goal:
    
            accuracy = 1
            
            break;

    return accuracy


def save_accuracy(k,accuracy):
    
    if k == 1:
            
        average_top1.append(accuracy)
        
    if k == 10:
            
        average_top10.append(accuracy)
        
    if k == 100:
            
        average_top100.append(accuracy)
        
    if k == 1000:
            
        average_top1000.append(accuracy)


average_top1 = []
average_top10 = []
average_top100 = []
average_top1000 = []

top_k = [1,10,100,1000]

for i in tqdm(range(len(article_dense_vector))):
    
    distance_vector = pairwise_distances(article_dense_vector[i].reshape(1,768), tables_dense_vector, metric='cosine')
    
    id_query_goal = int(articles_id[i])
    
    for accuracy_k in top_k:
        
        count_top_tables = accuracy_k
        
        top_k_rank = nsmallest(count_top_tables, distance_vector[0])
    
        id_ranked_tables = get_id_ranked_tables(top_k_rank,distance_vector[0])
        
        accuracy_value = get_accuracy(id_ranked_tables,id_query_goal)
        
        #save the accuracy on the list
        save_accuracy(accuracy_k,accuracy_value)


print(str(round(np.mean(average_top1),4))+" (±) "+str(round(np.std(average_top1),4)))
print(str(round(np.mean(average_top10),4))+" (±) "+str(round(np.std(average_top10),4)))
print(str(round(np.mean(average_top100),4))+" (±) "+str(round(np.std(average_top100),4)))
print(str(round(np.mean(average_top1000),4))+" (±) "+str(round(np.std(average_top1000),4)))
