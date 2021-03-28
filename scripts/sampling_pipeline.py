import torch
import scripts.model as mdl

class StoryPipeline:
    def __init__(self, model_path, bpe, embedding_dim, hidden_dim):
        self.bpe = bpe

        device = torch.device('cpu')
        state_dict = torch.load(model_path, map_location=device)

        self.model = mdl.SherlockLanguageModel(
            num_embeddings=len(bpe.tokens),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim
        )
        self.model.load_state_dict(state_dict["LM"])

        self.token2idx = tokens = dict(zip(bpe.tokens, range(len(bpe.tokens))))
        self.idx2tokens = dict(zip(range(len(self.bpe.tokens)), self.bpe.tokens))


    def __call__(self, initial_passage, temperature, num_tokens):
        encoded_passage = self._preprocess(initial_passage)
        story = self.model.sample_story(encoded_passage, temperature, num_tokens)

        return self._postprocess(story)

    def _preprocess(self, text):
        encoded_passage = [self.bpe.encode(word) for word in text.split()]
        encoded_passage = [item for sublist in encoded_passage for item in sublist]

        encoded_passage = [self.token2idx[story_token] for story_token in encoded_passage]

        return encoded_passage

    def _postprocess(self, sampled_story):
        story = [self.idx2tokens[token_index] for token_index in sampled_story]

        story = ' '.join(story)
        story = story.split("</w>")
        story = [word.replace(" ", "") for word in story]

        return ' '.join(story)


if __name__ == "__main__":
    import scripts.data_preprocessing as dp

    bpe = dp.BytePairEncoding.load("./assets/tokens_1k.json", "./assets/vocab_1k.json")
    pipeline = StoryPipeline(
        model_path='./models/sw16_epoch30_LM_GPU_e512_h1028.pth',
        bpe=bpe,
        embedding_dim=512,
        hidden_dim=1028
    )

    print(pipeline("Wiktor believe that", 0.7, 100))
