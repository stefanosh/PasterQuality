import numpy as np
from datetime import datetime, timedelta
import pandas as pd


def check_common_ids(train_ids, val_ids, test_ids, unique_ids):
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(unique_ids)
    is_mutually_exclusive = True

    # Check if train_ids, val_ids, and test_ids are mutually exclusive
    for train_id in train_ids:
        if train_id in val_ids or train_id in test_ids:
            is_mutually_exclusive = False
            break

    if is_mutually_exclusive:
        for val_id in val_ids:
            if val_id in train_ids or val_id in test_ids:
                is_mutually_exclusive = False
                break

    if is_mutually_exclusive:
        for test_id in test_ids:
            if test_id in train_ids or test_id in val_ids:
                is_mutually_exclusive = False
                break

    if is_mutually_exclusive:
        print("The train_ids, val_ids, and test_ids are mutually exclusive.")
    else:
        print("There are common ids among train_ids, val_ids, and test_ids.")