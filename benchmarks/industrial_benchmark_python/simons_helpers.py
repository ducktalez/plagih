from pathlib import Path
import pandas as pd
import numpy as np
from pathlib import Path


def make_labellist_tree_files():
    """
    too lazy to make them by hand
    """
    for ii, labelz in enumerate(['label_list,+,*,1.25,-12.31,Velocity_0',
                                 'label_list,+,*,-0.53,-29.91,Gain_0',
                                 'label_list,+,*,0.55,-29.22,Shift_0',
                                 'label_list,*,-12.31,Velocity_0',
                                 'label_list,*,-29.91,Gain_0',
                                 'label_list,*,-29.22,Shift_0']):
        with Path.open(Path('tree_labels-{}.csv'.format(ii)), 'w+') as file:
            file.write(labelz)


def samples_preprocessing_csv():
    samples_all = Path('../run_sources/IB/samples.csv')
    samples_prepared = Path('../run_sources/IB/samples_prepared.csv')
    df = pd.read_csv(samples_all)

    drop_cols = [column for column in df.columns if (column[-2] != '_' and column[0] != 'a') or 'Reward' in column or ('SetPoint' in column and column != 'SetPoint')]
    df = df.drop(drop_cols, axis=1)
    df = df.rename(columns={'SetPoint_0': 'SetPoint'})
    df.to_csv(samples_prepared, index=False)


samples_preprocessing_csv()
