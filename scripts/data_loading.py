import json

import torch


class SherlockDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, story_window, token_json):
        self.story_window = story_window

        with open(dataset_path, "r") as dset_file:
            self.dataset = dset_file.read().split()

        with open(token_json, "r") as json_file:
            self.tokens = json.load(json_file)
            self.tokens = dict(zip(self.tokens, range(len(self.tokens))))

    def __len__(self):
        return len(self.dataset) - self.story_window - 1

    def __getitem__(self, idx):
        story = self.dataset[idx : idx + self.story_window].copy()
        target = self.dataset[idx + self.story_window]

        story = self._one_hot_encoding(story)
        target = self._one_hot_encoding(target)

        return torch.tensor(story), torch.tensor(target)

    def _one_hot_encoding(self, tokens):
        if isinstance(tokens, str):
            tokens = [tokens]

        return [self.tokens[story_token] for story_token in tokens]
