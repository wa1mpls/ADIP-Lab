import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, classes, dropout=0.1):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim // 2)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embed_dim // 2, classes)

    def forward(self, inp):
        x = self.fc1(inp)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x