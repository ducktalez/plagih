import numpy as np

def tree_label_insert_list(xtype, depth_goal):
    """
    build a random tree based on a base depth and 50% chance for every node to become a terminal
    """
    todo_xtypes = [xtype]
    result_label_list = []
    result_arity_list = []
    result_type_list = []
    next_xtype_list = []

    # Build a list with labels in row, and a list with their arities
    for depth in range(0, depth_goal):
        next_xtype_list = []
        if depth == depth_goal - 1:  # now, we are on the lowest level.
            for t in todo_xtypes:  # Build terminals now.
                label = '2'
                arity = 0

                # Add the label to the result list
                result_label_list.append(label)
                result_arity_list.append(arity)
                result_type_list.append('term')

            print('### Lowest level', label)
            break
        else:
            for t in todo_xtypes:

                if np.random.choice(['fun', 'trm']) == 'trm':
                    label = '3'
                    arity = 0
                    xtype_child = '2f'
                else:
                    label, arity = '+', 2

                    if label == 'Ifte':
                        next_xtype_list.extend(['2b', '2f', '2f'])
                    else:
                        tmp_xtype = 'b2f'

                        xtype_child = tmp_xtype[:2][::-1]
                        for _ in range(0, arity):  # when arity==2, add 2 times
                            next_xtype_list.append(xtype_child)

                # Add the label to the result list
                result_label_list.append(label)
                result_arity_list.append(arity)
                result_type_list.append('idk')

            # Finally, update the list for the next round
            todo_xtypes = next_xtype_list[:]
            next_xtype_list = []

            print('new todos', len(todo_xtypes), 'label_list', result_label_list)

    print('Result_label_list', result_label_list, 'result_arity_list', result_arity_list)
    return result_label_list, result_arity_list, result_type_list

tree_label_insert_list('2f', 3)
