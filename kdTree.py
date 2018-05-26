from collections import namedtuple
from operator import itemgetter
from pprint import pformat
import itertools
import math
class Node(namedtuple('Node',"parent left_child right_child")):
    def __repr__(self):
        return pformat(tuple(self))

def create_kdTree(points_list, depth=0):
    global k
    try:
        k = len(points_list[0])
    except IndexError as e:
        return None

    axis = depth % k
    points_list.sort(key=itemgetter(axis))
    median = len(points_list) //2

    return Node(
        parent = points_list[median],
        left_child = create_kdTree(points_list[:median],depth+1),
        right_child = create_kdTree(points_list[median+1:],depth+1)
    )

def distance(a, b):
    s = 0
    for x, y in itertools.izip(a, b):
        d = x - y
        s += d * d
    return s

def closer_distance(pivot, p1, p2):
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    d1 = distance(pivot, p1)
    d2 = distance(pivot, p2)
    if d1 < d2:
        return p1
    else:
        return p2

def knn_search(tree, destination, depth=0):
    if tree is None:
        return None
    axis = depth % k
    next_branch = None
    opposite_branch = None
    # look for left-child and right-child
    if destination[axis] < tree.parent[axis]:
        next_branch = tree.left_child
        opposite_branch = tree.right_child
    else:
        next_branch = tree.right_child
        opposite_branch = tree.left_child

    # back to find the best result
    best = float('inf')
    best = closer_distance(destination,knn_search(next_branch,destination,depth + 1),tree.parent)

    if distance(destination, best) > abs(destination[axis] -tree.parent[axis]):
        best = closer_distance(destination,knn_search(opposite_branch,destination,depth + 1),best)

    return best


def main():
    point_list = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
    tree = create_kdTree(point_list)
    print tree

    point = (3,4.5)
    best = knn_search(tree,point)
    print best


if __name__ == '__main__':
    main()