import pandas as pd
import numpy as np
from pathlib import Path

samples_all = Path('gp_files/samples_raw.csv')  # Path('benchmarks/ib/gp_files/samples_raw.csv')
samples_prepared = Path('gp_files/samples_prepared.csv')


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
    df = pd.read_csv(samples_all)

    drop_cols = [column for column in df.columns if (column[-2] != '_' and column[0] != 'a') or 'Reward' in column or ('SetPoint' in column and column != 'SetPoint_0')]
    df = df.drop(drop_cols, axis=1)
    df = df.rename(columns={'SetPoint_0': 'SetPoint'})
    df.to_csv(samples_prepared, index=False)


def back_this_thing_up():
    df = pd.read_csv(samples_all)
    ppp = []
    for row_i in range(0, 10000, 100):
        ppp.append(df['SetPoint_0'][row_i])

    toreal = lambda x: (x * 28.72) + 55
    ppp = [int(round(toreal(p))) for p in ppp]
    unique, counts = np.unique(ppp, return_counts=True)
    counts = dict(zip(unique, counts))
    print(counts)
    conuts_dict_copypaste = {10: 11, 20: 7, 30: 16, 40: 10, 50: 11, 60: 10, 70: 12, 80: 7, 90: 6, 100: 10}


back_this_thing_up()
# samples_preprocessing_csv()
