from collections import deque
from space import Box
import numpy as np

class Node:
    """
    Node in the variable resolution tree
    """
    def __init__(self, space, left=None, right=None):
        self.space = space
        self.left = left
        self.right = right
        self.vals = []
        self.qs = []
        
    def add(self, val, q, err_thres, n_thres):
        self.vals.append(val)
        self.qs.append(q)
        if self.var > err_thres or len(self) > n_thres:
            self.split(err_thres, n_thres)
            
    def sample(self, n=1):
        assert self.is_leaf, 'Only leaf node allows to sample'
        samples = np.random.randn(n) * np.sqrt(self.var) + self.mean
        return samples
        
    def split(self, err_thres, n_thres):
        # find the dimension with the largest size
        size = self.space.high - self.space.low
        dim = np.argmax(size)
        # split halves in that dimension
        low = self.space.low
        high = self.space.high
        mid = (high[dim] + low[dim]) / 2
        high_left = high.copy()
        high_left[dim] = mid
        space_left = Box(low, high_left)
        low_right = low.copy()
        low_right[dim] = mid
        space_right = Box(low_right, high)
        
        self.left = Node(space_left)
        self.right = Node(space_right)
        
        for val, q in zip(self.vals, self.qs):
            if self.left.space.contains(val):
                self.left._add(val, q)
            else:
                self.right._add(val, q)
        
    def _add(self, val, q):
        self.vals.append(val)
        self.qs.append(q)
        
    @property
    def is_leaf(self):
        return (self.left is None) and (self.right is None)
    
    @property
    def mean(self):
        return np.asarray(self.qs).mean()
    
    @property
    def var(self):
        return np.asarray(self.qs).var()
    
    def __len__(self):
        return len(self.qs)
    
    def __repr__(self):
        return 'Node with {} observations in space [{},{}]'.format(len(self), self.space.low, self.space.high)
    
    def __str__(self):
        return self.__repr__()
        
class VRTree:
    """
    variable resolution tree
    """
    def __init__(self, space, err_thres, n_thres):
        self.root = Node(space)
        self.err_thres = err_thres
        self.n_thres = n_thres
        
    def observe(self, state, action, q):
        val = np.concatenate((state,action))
        self._add(self.root, val, q, self.err_thres, self.n_thres)
           
    def _add(self, node, val, q, err_thres, n_thres):
        if node.is_leaf:
            node.add(val, q, err_thres, n_thres)
        else:
            if node.left and node.left.space.contains(val):
                self._add(node.left, val, q, err_thres, n_thres)
            elif node.right and node.right.space.contains(val):
                self._add(node.right, val, q, err_thres, n_thres)
            else:
                raise ValueError('val: {} is out of bound'.format(val))
                
    def predict(self, state, action, n=1):
        val = np.concatenate((state,action))
        return self._sample(self.root, val, n)
        
    def _sample(self, node, val, n):
        if node.is_leaf:
            return node.sample(n)
        else:
            if node.left and node.left.space.contains(val):
                return self._sample(node.left, val, n)
            elif node.right and node.right.space.contains(val):
                return self._sample(node.right, val, n)
            else:
                raise ValueError('val: {} is out of bound'.format(val))
                
    def __repr__(self):
        res = ''
        q = deque()
        q.append(self.root)
        while q:
            curNode = q.popleft()
            if curNode.is_leaf:
                res += str(curNode)
                res += '\n'
            else:
                if curNode.left:
                    q.append(curNode.left)
                if curNode.right:
                    q.append(curNode.right)
        return res
                
    def __str__(self):
        return self.__repr__()