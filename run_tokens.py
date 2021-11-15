import exp_contrastive_training as cont_train
import exp_classification as clas

from glob import glob
from os import path

configs = glob("configs/**/*.json", recursive=True)

# bce/triplet/contrastive
for config in configs:
    if not "tokens" in config:
        continue
    print("### CONFIG: ", config)
    if path.exists(config.replace(".json", "_metrics.json")):
        continue
    if "classification" in config:
        clas.main(config)
    else:
        cont_train.main(config)
