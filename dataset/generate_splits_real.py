import random
import os
from shutil import move
import argparse
import math

from trainer import set_seed_all
from dataset.data_parser import valid_real_img_depth_mask, valid_real_state

def get_file_index(file_path):
    return int(file_path.split(os.sep)[-1].split(".")[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser("./generate_splits_real.py")
    parser.add_argument(
        '--data', '-d',
        type=str,
        required=False,
        default="./MoSS-Real",
        help='Path to the dataset. Default is ./MoSS-Real',
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

    parser.add_argument(
        '--offset', '-o',
        type=int,
        required=False,
        default=0,
        help='offset to index. default is 0',
    )

    FLAGS, unparsed = parser.parse_known_args()
    DATASET_PATH = FLAGS.data
    train_ratio = FLAGS.train
    val_ratio = FLAGS.eval
    test_ratio = FLAGS.test

    set_seed_all(52013146)

    assert math.isclose(train_ratio + val_ratio + test_ratio, 1.0)
    assert train_ratio > 0 and val_ratio >= 0 and test_ratio >= 0

    imgs = [os.path.join(DATASET_PATH, "img", p) for p in sorted(os.listdir(os.path.join(DATASET_PATH, "img"))) if valid_real_img_depth_mask(p)]
    depths = [os.path.join(DATASET_PATH, "depth", p) for p in sorted(os.listdir(os.path.join(DATASET_PATH, "depth"))) if valid_real_img_depth_mask(p)]
    masks = [os.path.join(DATASET_PATH, "mask", p) for p in sorted(os.listdir(os.path.join(DATASET_PATH, "mask"))) if valid_real_img_depth_mask(p)]
    states = [os.path.join(DATASET_PATH, "state", p) for p in sorted(os.listdir(os.path.join(DATASET_PATH, "state"))) if valid_real_state(p)]
    
    assert len(imgs) == len(depths)
    assert len(depths) == len(masks)
    assert len(states) == len(imgs)

    os.makedirs(os.path.join(DATASET_PATH, "train", "img"))
    os.makedirs(os.path.join(DATASET_PATH, "train", "depth"))
    os.makedirs(os.path.join(DATASET_PATH, "train", "mask"))
    os.makedirs(os.path.join(DATASET_PATH, "train", "state"))

    os.makedirs(os.path.join(DATASET_PATH, "eval", "img"))
    os.makedirs(os.path.join(DATASET_PATH, "eval", "depth"))
    os.makedirs(os.path.join(DATASET_PATH, "eval", "mask"))
    os.makedirs(os.path.join(DATASET_PATH, "eval", "state"))

    os.makedirs(os.path.join(DATASET_PATH, "test", "img"))
    os.makedirs(os.path.join(DATASET_PATH, "test", "depth"))
    os.makedirs(os.path.join(DATASET_PATH, "test", "mask"))
    os.makedirs(os.path.join(DATASET_PATH, "test", "state"))

    total_num = len(imgs)

    train_idx = random.sample(range(0, total_num), int(train_ratio * total_num))
    for i in train_idx:
        file_idx = get_file_index(imgs[i])
        old_file_idx_str = str(file_idx).zfill(6)
        new_file_idx_str = str(file_idx + FLAGS.offset).zfill(6)
        
        move(imgs[i], imgs[i].replace(DATASET_PATH, DATASET_PATH + "/train/").replace(old_file_idx_str, new_file_idx_str))
        move(depths[i], depths[i].replace(DATASET_PATH, DATASET_PATH + "/train/").replace(old_file_idx_str, new_file_idx_str))
        move(masks[i], masks[i].replace(DATASET_PATH, DATASET_PATH + "/train/").replace(old_file_idx_str, new_file_idx_str))
        move(states[i], states[i].replace(DATASET_PATH, DATASET_PATH + "/train/").replace(old_file_idx_str, new_file_idx_str))

    val_test_idx = [i for i in range(0, total_num) if i not in train_idx]
    total_val_test = len(val_test_idx)
    val_idx = []
    if val_ratio > 0:
        val_ratio_in_val_test = val_ratio / (val_ratio + test_ratio)
        val_idx = random.sample(val_test_idx, int(val_ratio_in_val_test * total_val_test))
        for i in val_idx:
            file_idx = get_file_index(imgs[i])
            old_file_idx_str = str(file_idx).zfill(6)
            new_file_idx_str = str(file_idx + FLAGS.offset).zfill(6)
            move(imgs[i], imgs[i].replace(DATASET_PATH, DATASET_PATH + "/eval/").replace(old_file_idx_str, new_file_idx_str))
            move(depths[i], depths[i].replace(DATASET_PATH, DATASET_PATH + "/eval/").replace(old_file_idx_str, new_file_idx_str))
            move(masks[i], masks[i].replace(DATASET_PATH, DATASET_PATH + "/eval/").replace(old_file_idx_str, new_file_idx_str))
            move(states[i], states[i].replace(DATASET_PATH, DATASET_PATH + "/eval/").replace(old_file_idx_str, new_file_idx_str))
    test_idx = [i for i in val_test_idx if i not in val_idx]
    for i in test_idx:
        file_idx = get_file_index(imgs[i])
        old_file_idx_str = str(file_idx).zfill(6)
        new_file_idx_str = str(file_idx + FLAGS.offset).zfill(6)
        move(imgs[i], imgs[i].replace(DATASET_PATH, DATASET_PATH + "/test/").replace(old_file_idx_str, new_file_idx_str))
        move(depths[i], depths[i].replace(DATASET_PATH, DATASET_PATH + "/test/").replace(old_file_idx_str, new_file_idx_str))
        move(masks[i], masks[i].replace(DATASET_PATH, DATASET_PATH + "/test/").replace(old_file_idx_str, new_file_idx_str))
        move(states[i], states[i].replace(DATASET_PATH, DATASET_PATH + "/test/").replace(old_file_idx_str, new_file_idx_str))
