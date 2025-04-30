from pathlib import Path
from litgpt.data import JsonChatDataModule

train_file = Path("../1_data_preprocessing/dataset/culture_wvs/train_dialogues.jsonl")
val_file = Path("../1_data_preprocessing/dataset/culture_wvs/val_dialogues.jsonl")

data_module = JsonChatDataModule(
    train_file=train_file,
    val_file=val_file,
    batch_size=2,
)
