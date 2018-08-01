import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
#         self.dropout = nn.Dropout(0.4)
#         self.hidden = self.init_hidden(num_layers, batch_size, hidden_size)
        
#     def init_hidden(self, num_layers, batch_size, hidden_size):
#         return (torch.zeros(num_layers, batch_size, hidden_size).to('cuda'),
#                 torch.zeros(num_layers, batch_size, hidden_size).to('cuda'))
    
    def forward(self, features, captions):
#         print(features.shape)
#         print(captions.shape)
#         print('===================')
        features = features.view(features.shape[0],1,features.shape[1])
        captions = captions[:,:-1]
        embedding = self.embedding(captions)
#         print(features.shape)
#         print(embed.shape)
        embedding = torch.cat([features, embedding], 1)
#         print(embed.shape)
        lstm_out,_ = self.lstm(embedding)
    
#         x = self.dropout(lstm_out)
#         print(lstm_out.shape)
        out = self.fc(lstm_out)
#         tag_scores = F.log_softmax(out, dim=2)
        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        result = []
        for i in range(max_len):
            output, states = self.lstm(inputs, states)
            output = self.fc(output.squeeze(1))
            index = output.max(1)[1]
            result.append(index.item())
            inputs = self.embedding(index).unsqueeze(1)
        
        return result
        
