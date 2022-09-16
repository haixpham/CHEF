import nltk
from nltk.draw.tree import TreeView
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import pathlib as plb
import svgling as svgl
import re
from collections import deque

VERBS = None
Lemmatizer = None

def tokenize(sentence, remove_period=False):
    tokens = sentence.split(" ") #nltk.word_tokenize(sentence)
    if remove_period:
        # iterate through all of tokens, remove period from all tokens
        tokens = [word.strip(".") for word in tokens]
    tokens = list(filter(None, tokens))
    return tokens

def get_verb_set():
    verbs = {x.name().split('.', 1)[0] for x in wordnet.all_synsets('v')} # list of English verbs
    verbs.add("peel")
    return verbs


def is_verb(word):
    global VERBS, Lemmatizer
    if VERBS is None:
        VERBS = get_verb_set()
    if Lemmatizer is None:
        Lemmatizer = WordNetLemmatizer()
    t = Lemmatizer.lemmatize(word.lower())
    return t in VERBS


def save_tree_to_image(tree, filename):
    """
    :param tree: nltk parse tree
    :param filename: PS file name
    :return: n/a
    """
    if not tree:
        print("tree not exist")
        return

    if plb.Path(filename).suffix != ".ps":
        filename = filename + ".ps"
    tv = TreeView(tree)
    tv._cframe.print_to_file(filename)
    tv.destroy()


def save_tree_to_image_and_latex(tree, image_name, text_name):
    if not tree:
        print("tree not exist")
        return

    save_tree_to_image(tree, image_name)
    with open(text_name, "w") as f:
        f.write(tree.pformat_latex_qtree())

def save_tree_to_svg(tree, filename):
    if not tree:
        print("tree not exist")
        return

    if plb.Path(filename).suffix != ".svg":
        filename = filename + ".svg"

    tree_layout = svgl.core.TreeLayout(tree)
    tree_svg = tree_layout.svg_build_tree()
    tree_svg.saveas(filename)


def save_tree_to_svg_and_latex(tree, image_name, text_name):
    if not tree:
        print("tree not exist")
        return

    save_tree_to_svg(tree, image_name)
    with open(text_name, "w") as f:
        f.write(tree.pformat_latex_qtree())

def save_tree_to_latex(tree, text_name):
    with open(text_name, "w") as f:
        f.write(tree.pformat_latex_qtree())

def save_tree_string(tree, filename):
    with open(filename, "w") as f:
        f.write(str(tree))

def load_tree(tree_str):
    new_str = re.sub("\(-[0-9]+-\)", "o", tree_str)
    tree = nltk.tree.Tree.fromstring(new_str)
    return tree


def load_tree_from_text_file(filename):
    if plb.Path(filename).exists():
        tree_str = ""
        with open(filename) as f:
            for line in f:
                tree_str = tree_str + " " + line.rstrip("\n").strip()
        return load_tree(tree_str)
    else:
        return None


def get_leaves(tree):
    leaves = tree.leaves()
    ret = {}
    for i, leaf in enumerate(leaves):
        ret[leaf] = len(tree.leaf_treeposition(i))
    return ret


def find_lowest_leaves(tree):
    leaves = get_leaves(tree)
    min_depth = 100
    ret = []
    for le in leaves:
        if leaves[le] < min_depth:
            ret = [le]
            min_depth = leaves[le]
        elif leaves[le] == min_depth:
            ret.append(le)
    return ret, min_depth


def find_highest_leaves(tree):
    leaves = get_leaves(tree)
    max_depth = 0
    ret = []
    for le in leaves:
        if leaves[le] > max_depth:
            ret = [le]
            max_depth = leaves[le]
        elif leaves[le] == max_depth:
            ret.append(le)
    return ret, max_depth


"""
def remove_parent(tree, parent_treeposition):
    parent_of_parent = tree[parent_treeposition].parent()
    n = len(parent_of_parent)
    del tree[parent_treeposition]
    if len(parent_of_parent) == 0:
        parent_of_parent_position = parent_of_parent.treeposition()
        remove_parent(tree, parent_of_parent_position)
    else:
        return
"""

def remove_leaf(tree, leaf):
    leaves = tree.leaves()
    leaf_index = leaves.index(leaf)
    leaf_treeposition = tree.leaf_treeposition(leaf_index)

    #tree[leaf_treeposition] = None
    # check if the parent node does not have any children, then remove the parent, recursively

    parent_node = tree[leaf_treeposition[:-1]]
    if len(parent_node) == 1:
        # delete parent
        parent_treeposition = parent_node.treeposition()
        parent_of_parent = tree[parent_treeposition[:-1]]
        while len(parent_of_parent) == 1 and parent_of_parent.parent() is not None:
            parent_of_parent_treeposition = parent_of_parent.treeposition()
            new_parent_of_parent = tree[parent_of_parent_treeposition[:-1]]
            if len(new_parent_of_parent) != 1:
                break
            else:
                parent_of_parent = new_parent_of_parent
        if len(parent_of_parent) == 1:
            parent_of_parent_treeposition = parent_of_parent.treeposition()
            del tree[parent_of_parent_treeposition]
        else:
            del tree[parent_treeposition]
    else:
        del tree[leaf_treeposition]



def rebalance_tree_recursive(tree_root, subtree):
    if not isinstance(subtree, nltk.tree.Tree):
        return
    if len(subtree) == 1:
        tree_position = subtree.treeposition()
        parent = subtree.parent()
        parent_index = subtree.parent_index()
        first_child = None
        for child in subtree:
            if isinstance(child, nltk.tree.Tree):
                first_child = child.copy(deep=True) #must be deepcopy, because if we del the parent, the child is also deleted
            else:
                first_child = child
            break
        del tree_root[tree_position]
        parent.insert(parent_index, first_child)
        if isinstance(first_child, nltk.tree.Tree):
            rebalance_tree_recursive(tree_root, first_child)
    elif len(subtree) == 0:
        # delete it
        tree_position = subtree.treeposition()
        del tree_root[tree_position]
    else:
        for child in subtree:
            if isinstance(child, nltk.tree.Tree):
                rebalance_tree_recursive(tree_root, child)


def rebalance_tree(tree_root):
    if len(tree_root) < 2:
        return
    for child in tree_root:
        rebalance_tree_recursive(tree_root, child)

