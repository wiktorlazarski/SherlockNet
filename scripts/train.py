import argparse
import logging
import math
import os

import torch
from torch.utils import data

import scripts.data_loading as dl
import scripts.model as mdl

ENCODED_WITH_BPE_DATASET_PATH = "./data/encoded_corpus.txt"
BPE_TOKENS_PATH = "./assets/tokens_1k.json"


def train(num_epochs, story_window, batch_size, lr, embedding_dim, hidden_dim, dropout, checkpoint=None):
    checkpoint = torch.load(checkpoint) if checkpoint is not None else None

    dset = dl.SherlockDataset(
        ENCODED_WITH_BPE_DATASET_PATH,
        story_window,
        BPE_TOKENS_PATH
    )

    data_loader = data.DataLoader(
        dataset=dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=math.ceil(os.cpu_count() / 2)
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training on device {torch.cuda.get_device_name(device.index)}")

    sherlock_lm = mdl.SherlockLanguageModel(
        num_embeddings=len(dset.tokens),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    )
    if checkpoint is not None:
        sherlock_lm.load_state_dict(checkpoint["LM"])
    sherlock_lm.to(device)

    optimizer = torch.optim.Adam(params=sherlock_lm.parameters(), lr=lr)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optim"])

    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)

    start_epoch = 1 if checkpoint is None else checkpoint["epoch"] + 1
    for epoch in range(start_epoch, start_epoch + num_epochs):
        cost = 0.0
        running_loss = 0.0

        for step, batch in enumerate(data_loader, 1):
            text, label = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()

            preds = sherlock_lm.forward(text)

            loss = criterion(preds, label)

            loss.backward()
            optimizer.step()

            cost += loss.item()
            running_loss += loss.item()

            every_step = 100
            if not step % every_step:
                avg_loss = running_loss / every_step
                running_loss = 0.0
                logging.info(f"Epoch {epoch} Step {step}/{len(data_loader)} => {avg_loss: .4f}")

        logging.info(f"Epoch {epoch} Cost Function => {cost / len(data_loader): .4f}")

        torch.save(
            {
                'epoch': epoch,
                'optim': optimizer.state_dict(),
                'LM': sherlock_lm.state_dict()
            },
            f"./models/epoch{epoch}_LM_GPU_e{embedding_dim}_h{hidden_dim}.pth"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image Captioning with Visual Attention training process")

    parser.add_argument("--num_epochs", default=5, type=int, help="Number of epochs to perform in training process")
    parser.add_argument("--story_window", default=128, type=int, help="Number of tokens considered when training")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size")
    parser.add_argument("--lr", default=3e-4, type=float, help="Learning rate. If checkpoint passed then learning rate will be loaded from state_dict")
    parser.add_argument("--embedding_dim", default=256, type=int, help="Word embedding dimmension")
    parser.add_argument("--hidden_dim", default=512, type=int, help="LSTM layer dimmension")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout regularization")
    parser.add_argument("--checkpoint_path", type=str, help="Checkpoint path")

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d/%m/%Y %H:%M")

    args = parse_args()

    train(
        num_epochs=args.num_epochs,
        story_window=args.story_window,
        batch_size=args.batch_size,
        lr=args.lr,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        checkpoint=args.checkpoint_path
    )
