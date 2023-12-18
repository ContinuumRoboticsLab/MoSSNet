import random
import os
from shutil import move
import argparse
import math

from trainer import set_seed_all
from dataset.data_parser import valid_depth, valid_img, valid_meta

if __name__ == "__main__":
    parser = argparse.ArgumentParser("./generate_splits.py")
    parser.add_argument(
        '--data', '-d',
        type=str,
        required=False,
        default="./MoSS-Sim",
        help='Path to the dataset. Default is ./MoSS-Sim',
    )
    parser.add_argument(
        '--train', '-r',
        type=float,
        required=False,
        default=0.6,
        help='Training set ratio. Default is 0.6',
    )

    parser.add_argument(
        '--eval', '-v',
        type=float,
        required=False,
        default=0.15,
        help='Val set ratio. Default is 0.15',
    )

    parser.add_argument(
        '--test', '-t',
        type=float,
        required=False,
        default=0.25,
        help='Test set ratio. Default is 0.25',
    )

    FLAGS, unparsed = parser.parse_known_args()
    DATASET_PATH = FLAGS.data
    train_ratio = FLAGS.train
    val_ratio = FLAGS.eval
    test_ratio = FLAGS.test

    set_seed_all(52013146)

    assert math.isclose(train_ratio + val_ratio + test_ratio, 1.0)
    assert train_ratio > 0 and val_ratio >= 0 and test_ratio >= 0

    imgs = [os.path.join(DATASET_PATH, "img", p) for p in sorted(os.listdir(os.path.join(DATASET_PATH, "img"))) if valid_img(p)]
    depths = [os.path.join(DATASET_PATH, "depth", p) for p in sorted(os.listdir(os.path.join(DATASET_PATH, "depth"))) if valid_depth(p)]
    metas = [os.path.join(DATASET_PATH, "meta", p) for p in sorted(os.listdir(os.path.join(DATASET_PATH, "meta"))) if valid_meta(p)]
    states = [os.path.join(DATASET_PATH, "state", os.path.split(p)[-1].split("_")[0] + ".csv") 
        for p in imgs if os.path.exists(os.path.join(DATASET_PATH, "state", os.path.split(p)[-1].split("_")[0] + ".csv"))]
    
    assert len(imgs) == len(depths)
    assert len(depths) == len(metas)
    assert len(states) == len(imgs)

    os.makedirs(os.path.join(DATASET_PATH, "train", "img"))
    os.makedirs(os.path.join(DATASET_PATH, "train", "depth"))
    os.makedirs(os.path.join(DATASET_PATH, "train", "meta"))
    os.makedirs(os.path.join(DATASET_PATH, "train", "state"))

    os.makedirs(os.path.join(DATASET_PATH, "eval", "img"))
    os.makedirs(os.path.join(DATASET_PATH, "eval", "depth"))
    os.makedirs(os.path.join(DATASET_PATH, "eval", "meta"))
    os.makedirs(os.path.join(DATASET_PATH, "eval", "state"))

    os.makedirs(os.path.join(DATASET_PATH, "test", "img"))
    os.makedirs(os.path.join(DATASET_PATH, "test", "depth"))
    os.makedirs(os.path.join(DATASET_PATH, "test", "meta"))
    os.makedirs(os.path.join(DATASET_PATH, "test", "state"))

    total_num = len(imgs)

    train_idx = random.sample(range(0, total_num), int(train_ratio * total_num))
    for i in train_idx:
        move(imgs[i], imgs[i].replace(DATASET_PATH, DATASET_PATH + "/train/"))
        move(depths[i], depths[i].replace(DATASET_PATH, DATASET_PATH + "/train/"))
        move(metas[i], metas[i].replace(DATASET_PATH, DATASET_PATH + "/train/"))
        move(states[i], states[i].replace(DATASET_PATH, DATASET_PATH + "/train/"))
    
    val_test_idx = [i for i in range(0, total_num) if i not in train_idx]
    total_val_test = len(val_test_idx)
    val_idx = []
    if val_ratio > 0:
        val_ratio_in_val_test = val_ratio / (val_ratio + test_ratio)
        val_idx = random.sample(val_test_idx, int(val_ratio_in_val_test * total_val_test))
        for i in val_idx:
            move(imgs[i], imgs[i].replace(DATASET_PATH, DATASET_PATH + "/eval/"))
            move(depths[i], depths[i].replace(DATASET_PATH, DATASET_PATH + "/eval/"))
            move(metas[i], metas[i].replace(DATASET_PATH, DATASET_PATH + "/eval/"))
            move(states[i], states[i].replace(DATASET_PATH, DATASET_PATH + "/eval/"))
    test_idx = [i for i in val_test_idx if i not in val_idx]
    for i in test_idx:
        move(imgs[i], imgs[i].replace(DATASET_PATH, DATASET_PATH + "/test/"))
        move(depths[i], depths[i].replace(DATASET_PATH, DATASET_PATH + "/test/"))
        move(metas[i], metas[i].replace(DATASET_PATH, DATASET_PATH + "/test/"))
        move(states[i], states[i].replace(DATASET_PATH, DATASET_PATH + "/test/"))
