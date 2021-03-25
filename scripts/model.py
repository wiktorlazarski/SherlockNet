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

    def sample_story(self, passage, temperature, length, story_window=16):
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            generated_story = passage.copy()
            if len(generated_story) > story_window:
                passage = generated_story[-story_window:]

            for _ in range(length):
                print(len(passage))
                passage = torch.tensor(passage).unsqueeze(dim=0)

                preds = self.forward(passage)
                preds = torch.softmax(preds / temperature, dim=1)

                y_pred = torch.multinomial(preds, 1)

                new_token = y_pred.item()

                generated_story.append(new_token)

                if len(generated_story) > story_window:
                    passage = generated_story[-story_window:]
                else:
                    passage = generated_story

        return generated_story


if __name__ == "__main__":
    device = torch.device('cpu')
    state_dict = torch.load(
        './models/sw16_epoch25_LM_GPU_e512_h1028.pth',
        map_location=device
    )

    import scripts.data_preprocessing as dp
    bpe = dp.BytePairEncoding.load("./assets/tokens_1k.json", "./assets/vocab_1k.json")

    sherlock_lm = SherlockLanguageModel(
        num_embeddings=len(bpe.tokens),
        embedding_dim=512,
        hidden_dim=1028
    )
    sherlock_lm.load_state_dict(state_dict["LM"])
    print("Model loaded successfully")

    init_passage = "Wiktor was a Sherlock friend who helped him when"

    encoded_passage = [bpe.encode(word) for word in init_passage.split()]
    encoded_passage = [item for sublist in encoded_passage for item in sublist]

    tokens = dict(zip(bpe.tokens, range(len(bpe.tokens))))
    encoded_passage = [tokens[story_token] for story_token in encoded_passage]


    temperature = 0.005
    story = sherlock_lm.sample_story(encoded_passage, temperature, 100)
    print("Sampling finished")

    tokens = dict(zip(range(len(bpe.tokens)), bpe.tokens))
    story = [tokens[token_index] for token_index in story]

    story = ' '.join(story)
    story = story.split("</w>")
    story = [word.replace(" ", "") for word in story]

    story = ' '.join(story)
    print(story)
