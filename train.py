import argparse
from shutil import copyfile
from pathlib import Path
from typing import Literal

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from cv_training import validate

from utils.data_module import PlagiarismFunctions
from utils.lightning_module import PlagiarismModel

seed_everything(420)


graph_architecture = {
    "embedding_dim": 4,
    "hidden_dim": 64,
    "num_layers": 1,
    "residual": False
}

tokens_architecture = {
    "embedding_dim": 8,
    "hidden_dim": 128,
    "num_layers": 3
}

data_cache = "data/cache"

def main(format: Literal["graph", "tokens"], classify: bool = False):
    if format == "tokens":
        data_root = Path("data/functions")
        batch_size = 50
        model_params = tokens_architecture
        model_type = "lstm"
    else:
        data_root = Path("data/graph_functions")
        batch_size = 256
        model_params = graph_architecture
        model_type = "gnn"

    train_split = data_root / "train.txt"
    with train_split.open() as f:
        train_functions = list(map(lambda l: l.strip(), f.readlines()))

    test_split = data_root / "train.txt"
    with test_split.open() as f:
        test_functions = list(map(lambda l: l.strip(), f.readlines()))


    data_params = {
        "root_dir": str(data_root),
        "format": format,
        "cache": data_cache,
        "batch_size": batch_size
    }

    data_module = PlagiarismFunctions(classify, train_functions, test_functions, **data_params)
    data_module.setup()

    model = PlagiarismModel(classify, **model_params, model_type=model_type, num_tokens=data_module.num_features)
    checkpoint_callback = ModelCheckpoint(monitor='loss/val', save_top_k=1)
    early_stopping = EarlyStopping("loss/val", patience=10)
    trainer = Trainer(gpus=1, max_epochs=100, callbacks=[checkpoint_callback, early_stopping], num_sanity_val_steps=0)

    trainer.fit(model, train_dataloader=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())

    dst_model_path = Path("best")
    dst_model_path.mkdir(exist_ok=True)
    dst_model_path = dst_model_path / f"{model_type}_{'_'.join(map(str, list(model_params.values()) + (['classifier'] if classify else ['encoder'])))}.ckpt"
    copyfile(checkpoint_callback.best_model_path, str(dst_model_path))

    model = PlagiarismModel.load_from_checkpoint(checkpoint_callback.best_model_path)

    roc_auc = validate(model, data_module.test_dataloader())
    print(f"ROC AUC: {roc_auc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str, choices=["graph", "tokens"])
    parser.add_argument("-c", action="store_true")
    args = parser.parse_args()
    main(args.f, args.c)
