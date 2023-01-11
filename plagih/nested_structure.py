from dataclasses import dataclass
import itertools


@dataclass
class NestedStruc:
    """
    The core is the structure of a plagih gp-fintree.
    It recursively holds the nodes of a fintree; every fintree has a list of potential children.
    Example core: [+, 1, [*, [-, 2, 3], 2]] = 1 + ((2-3) * 2)
    states? sfeh:discuss
    [None]: not set
    [0]:    evolution/construction/build mode (potentially missing leaf nodes)
    [1]:    structurally complete/finalized branch (node_depths correct, node_id set, ...)
    [2]:    root-correct structure
    [3]:    including meta-data (fitness_train, complexity)
    """

    def __init__(self, anylabel, depth=None, is_fix=False, childs=None):
        try:
            anylabel.args = None  # sfeh: or remove childs, isfix
            anylabel.is_fix = None
            self.label: 'NodeBase' = anylabel
        except AttributeError:
            self.label = anylabel

        self.is_fix = is_fix
        self.childs = childs or []
        self.depth = depth

    def __str__(self):
        """
        Printing the nodes as nested array structure, easy to read.
        Also, have a look at repr(self) for a more detailed result
        """
        label_str = self.get_nlabel()

        if self.childs:
            childstr = ', '.join([str(x) for x in self.childs])
            label_str = f"{label_str}, {childstr}"

        return f"[{label_str}]"

    def eval_str(self):

        return self.get_nlabel()  # sfeh open

    def __repr__(self):
        """
        Printing the nodes as nested array structure such that it can be saved/loaded
        very closely related to str(), but adds the following information:
        - ":fix", when nodes are fixed
        """
        label_str = self.get_nlabel()
        label_str = str(label_str)

        if self.is_fix:
            label_str += ':fix'

        if self.childs:
            childstr = ', '.join([repr(x) for x in self.childs])
            label_str = f"{label_str}, {childstr}"
        return f"[{label_str}]"

    # def choose_term(xtype_out, choose_obs, choose_distributions, precision):
    #
    #     # sfeh 50% chance observation/value
    #     if random.choice(['obs', 'distrib']) == 'obs' and choose_obs[xtype_out]:
    #         obs = choose_obs[xtype_out]()
    #         # prsint('SAME???', obs.name, obs.label)  # sfeh
    #         return obs
    #     else:
    #         dist_fun = random.choice(choose_distributions[xtype_out])
    #         value = dist_fun()
    #         if xtype_out == float:  # sfeh int aswell?
    #             value = float(round(value, precision))
    #             const = FloatConstant(value)
    #         elif xtype_out == bool:
    #             const = BoolConstant(value)
    #         else:
    #             raise Exception('ASDASD NOOO WHYY')
    #         return const

    def __len__(self):
        """
        counting the amount of nodes recursively
        """
        return 1 + sum([len(cc) for cc in self.childs])

    def get_label(self):
        return self.label

    def get_nlabel(self):
        return self.label

    def get_arity(self):
        return len(self.label.xtype[0])

    def get_xtype(self):
        return self.label.xtype

    def get_xtype_in(self):
        return self.label.xtype[0]

    def get_xtype_out(self):
        return self.label.xtype[1]

    def is_root(self):
        """
        does this work while building a fintree?
        """
        return self.depth == 0

    def set_label(self, label: 'NestedStruc'):
        """
        all other values are automatically set by assigning the respected node
        """
        self.label = label

    def update_fixed_nodes(self, origin: 'NestedStruc'):
        """
        Updating the fixed nodes in a tree where they were lost for some reason.
        This should never be the case! But it happened during development of recreating a tree from expression.
        This might also be useful in tree checks
        """
        if origin.is_fix:
            if str(self.label) != str(origin.label):
                raise
            self.is_fix = True
            for ii, cc in enumerate(self.childs):
                cc.update_fixed_nodes(origin.childs[ii])

    def get_nodes_to_depth(self, goal_depth, only_mutable=False, get_closest_depth=False):
        """
        sum_layers=False, get_closest=True, return_all_layers=False
        """
        child_results = []
        if self.depth < goal_depth:
            child_results = sum(
                [child.get_nodes_to_depth(goal_depth, only_mutable=only_mutable, force_depth=get_closest_depth) for
                 child in self.childs], [])

        if only_mutable and self.is_fix or \
                get_closest_depth and self.depth != goal_depth:
            my_result = []
        else:
            my_result = [self]

        return my_result + child_results

    def get_all_nodes(self):
        if len(self.childs) == 0:
            return [self]
        else:
            return [self] + [cc.get_all_nodes() for cc in self.childs]

    def get_nodes_at_depth(self, goal_depth, allow_fixed=False, expand_depth=False):
        """
        Returns a list with mutable ids which are *goal_depth* layers away from non-modifiable nodes
        last_leaves: if you want so save all leave nodes aswell
        sum_layers=False, get_closest=True, return_all_layers=False
        """
        nodes = []
        if (not self.is_fix or allow_fixed) and (
                self.depth == goal_depth or (self.depth > goal_depth and expand_depth)):
            return [self]
        elif self.depth <= goal_depth or expand_depth:
            nodes.extend(list(itertools.chain(
                *[cc.get_nodes_at_depth(goal_depth, allow_fixed=allow_fixed, expand_depth=expand_depth) for cc in
                  self.childs])))
            return nodes
        else:
            return []

    def eval_apted_notation(self):
        """
        Calculating the TED requires this (weird) representation
        """
        # sfEh check if this still works as one-liner
        return f"{{{self.get_nlabel()}{''.join([cc.eval_apted_notation() for cc in self.childs])}}}"

    def get_max_depth(self, depth=0):
        """
        Go through all nodes, save depth
        """
        if len(self.childs) == 0:
            return depth
        else:
            return max(cc.get_max_depth(depth=depth + 1) for cc in self.childs)

    def repair_depth(self, depth=0):
        """
        aka set_depth recursively for all nodes in a branch
        mainly used in branch
        The depth is written inevery node (for whatever reason), and instead of having to propagate
        the depth through every crossover/branch mutation function, instead, we call it when replacing nodes
        """
        self.depth = depth
        for cc in self.childs:
            cc.repair_depth(depth=depth + 1)

    def set_new_node(self, new_node: 'NestedStruc'):
        self.set_label(new_node)  # sfeh remove childs, is_fix...
        self.childs = new_node.childs or []  # maybe must be updated recursively
        self.repair_depth(self.depth)  # Especially required for crossover or branches

    def eval_mutable_nodes(self, xtype_out=None, allow_root=True):
        """
        return all nodes that are mutable (non fixed)
        sfeh: is returning nodes large overhead? eg in large trees? if it is, return nodepaths only!
        """
        node_list = []
        if not self.is_fix:  # requirement for mutability
            # crossover requires excluding types that are not matching, and excludes the root node
            # sfeh:open in anderer klasse
            if (xtype_out is None or xtype_out == self.get_xtype_out()) and (allow_root or not self.is_root()):
                node_list.append(self)

        for cc in self.childs:
            node_list.extend(cc.eval_mutable_nodes(xtype_out=xtype_out, allow_root=allow_root))
        # deprecated:
        # node_list.extend(list(itertools.chain(
        #     *[cc.eval_mutable_nodes(xtype_out=xtype_out, allow_root=allow_root) for cc in self.childs])))
        return node_list

    def evolve_mutate_filter_branch(self, precision=6):
        """
        Recursively filter the nodes in the branch of fintree
        sfeh:   random filter all terminal nodes /
                single node /
                nodes in a branch /
                random nodes in a branch /
                intelligent filtering
        """
        # self.state = STATE_BUILDING  #  ==>state
        if self.get_arity() > 0:
            for cc in self.childs:
                cc.evolve_mutate_filter_branch(precision=precision)
        else:
            # self.label.mutate_self_filter(filter_type='gaussian_filter', precision=precision)
            # sfeh:xxx
            pass

    def finalize_set_depth(self, depth=0, recursive=True):
        """
        depth=0 is the root node
        """
        self.depth = depth
        max_depth = depth
        if recursive:
            for cc in self.childs:
                cc_depth = cc.finalize_set_depth(depth=depth + 1)
                max_depth = max(cc_depth, max_depth)

        return max_depth

    def check_typing(self, xtype_parent, fatal=True):  # sfeh
        """
        Checks, if all child nodes match the parents nodes types
        """
        result = [self.get_xtype_out() == xtype_parent,
                  self.get_xtype_in() == tuple([cc.get_xtype_out() for cc in self.childs]),
                  len(self.childs) == self.get_arity()]

        if sum(result) < len(result) and fatal:
            raise  # sfeh

        for ii, cc in enumerate(self.childs):
            cc.check_typing(self.get_xtype_in()[ii], fatal=fatal)

        return True

    def check_depth_infos(self, depth=0, fatal=None):
        """
        Checks, whether the depth of all nodes in the tree are set correctly.
        Should not be necessary if all evolutions (branch-mutation, crossover) work fine
        - starts at depth 0, increase at every level
        - compare with the written value
        :return: boolean
        """
        if depth != self.depth:
            if fatal:
                raise
            return False
        return all([cc.check_depth_infos(depth=depth + 1, fatal=fatal) for cc in self.childs])

    def selfcheck(self, check_depth=True, fatal=None):
        """
        Tree Self-check for its structure
        """
        results = [self.check_depth_infos(fatal=fatal) if check_depth else 0]
        return sum(results)