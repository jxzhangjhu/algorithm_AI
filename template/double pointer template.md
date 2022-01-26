## 题目

https://leetcode.com/problems/minimum-size-subarray-sum/



## 相向双指针 - patition in quicksort 

```python
class Solution:
    def patition(self, A, start, end):
        if start >= end:
            return 

        left, right = start, end
        # key point 1: pivot is the value, not the index 
        pivot = A[(start + end) // 2]
        # key point 2: every time you compare left & right, it should be left <= right not left < right 
        while left <= right:
            while left <= right and A[left] < pivot:
                left += 1
            while left <= right and A[right] > pivot: 
                right -= 1
            if left <= right:
                A[left], A[right] = A[right], A[left]
                left += 1
                right -= 1
```

## 背向双指针 

```python
class Solution:
    def patition(self, A, start, end):

        left = position 
        right = position + 1
        while left >=0 and right < len(s):
            if left and right 可以停下来了：
                break

            left -= 1
            right += 1 


```

## 同向双指针 - 快慢指针 

```python
class Solution:
    def patition(self, A, start, end):

        j = 0
        for i in range(n):
            # 不满足则循环到满足搭配为止

            while j < n and i and j 之间不满足条件:
                j += 1

            if i到j之间满足条件:
                处理i到j这段区间 

```


## 合并双指针 

```python
class Solution:
    def merge(self, A, start, end):
        new_list = [] 
        i, j = 0, 0

        # 合作的过程只能操作i, j 的移动， 不要去list.pop(0) 之类的操作, 因为pop(0) 是O(n)的时间复杂度在python
        while i < len(list1) and j < len(list2):
            if list1[i] < list[j]:
                new_list.append(list1[i]) 
                i += 1
            else:
                new_list.append(list2[j])
                j += 1

            # 合并剩下的数到new_list里
            # 不要用new_list.extend(list1[i:])之类的方法
            # 因为list1[i:] 会产生额外的空间消耗

            while i < len(list1):
                new_list.append(list1[i])
                i += 1
            while j < len(list2):
                new_list.append(list2[j])
                j += 1

            return new_list 



```
