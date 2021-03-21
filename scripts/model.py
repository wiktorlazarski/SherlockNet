import torch
import torch.nn as nn

class SherlockLanguageModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, dropout=0.5):
        super().__init__()

        self.token_embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim)

        self.bi_gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True
        )

        self.dropout = nn.Dropout(p=dropout)

        self.out_layer = nn.Linear(
            in_features=2*hidden_dim,
            out_features=num_embeddings
        )

    def forward(self, data_batch):
        embeddings = self.token_embedding(data_batch)

        _, out = self.bi_gru(embeddings)
        out = torch.cat((out[0], out[1]), dim=1)

        out = self.out_layer(self.dropout(out))

        return out

    def sample_story(self, starting_passage):
        self.eval()

        with torch.no_grad():
            pass

        return
