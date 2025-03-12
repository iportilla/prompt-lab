import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def get_sentiment(sentences):
    bert_dict = {}
    vectors = tokenizer(sentences, padding = True, max_length = 65, return_tensors='pt').to(device)
    outputs = bert_model(**vectors).logits
    probs = torch.nn.functional.softmax(outputs, dim = 1)
    for prob in probs:
        bert_dict['neg'] = round(prob[0].item(), 3)
        bert_dict['neu'] = round(prob[1].item(), 3)
        bert_dict['pos'] = round(prob[2].item(), 3)
        print (bert_dict)

MODEL_NAME = 'RashidNLP/Finance-Sentiment-Classification'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels = 3).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

get_sentiment(["The stock market will struggle until debt ceiling is increased", "ChatGPT is boosting Microsoft's search engine market share"])
