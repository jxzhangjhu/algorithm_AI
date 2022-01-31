# heap 

## 定义
1. 分为最小堆minheap，和最大堆，maxheap， 也就是最小元素或者最大元素在堆顶
2. 堆是一个完全二叉树，但堆的底层实现一般是数组，而不是二叉树
3. 孩子节点都比父亲节点大， 但是左右孩子的大小不影响
4. 堆不是binary search tree 
5. 堆的操作是从上到下， 从左到右

## 基本操作 - 高度是logn 
1. 构建堆 heapify - o(n)
2. 遍历堆 o(nlogn)
3. add - o(n)
4. remove - 理论上是o(logn) 但实际上python的库函数是for loop遍历的，所以是o(n)
5. pop - o(logn)
6. min or max - o(1) 
7. 由于是数组操作，选定k，父亲是k/2，左孩子kx2, 右孩子kx2+1
8. 可以结合hashmap 去查询或者remove 指定值 


## 使用条件
1. 找最大值或者最小值（60%）
2. 找第k大（pop k 次 复杂度o(nlogk)） (50%)
3. 要求logn时间对数据进行操作 （40%）  

##堆不能解决的问题
1. 查询比某个数大的最小值或者最近接的值 (平衡二叉树 balanced bst 才可以解决)
2. 找某段区间的最大值最小值 （线段树segmenttree 可以解决）
3. o(n) 找第k大的数 （需要使用快排中的partition操作）


## time complexity
## space complexity


## template 
```python

from heapq import heappush, heappop

class Heap:
    def __init__(self):
        self.minheap()
        self.deleted_set = set()

    def push(self, index, val):
        heappush(self.minheal, (val, index))

    def _lazy_deletion(self):
        while self.minheap and self.minheap[0][1] in self.deleted_set:
            heappop(self.minheap)

    def top(self):
        self._lazy_deletion()
        return self.minheap[0]

    def pop(self):
        self._lazy_deletion()
        heappop(self.minheap)

    def delete(self, index):
        self.deleted_set.add(index)

    def is_empty(self):
        return not bool(self.minheap)

```

## 几种类型的双指针及相关题目





