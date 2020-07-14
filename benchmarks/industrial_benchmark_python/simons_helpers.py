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


def help_make_programs(pareto_line):
    pass