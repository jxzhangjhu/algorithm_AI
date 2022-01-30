## 使用条件
1. 排序数组（30%-40% 是二分法）
2. 当面试官要求找一个比o(n)更小的时间复杂度算法的时候，99% 就是二分logn
3. 找到数组中的一个分割位置，使得左半部分满足某个条件，右半部分不满足 100% 就是二分
4. 找到一个最大、最小的值使得某个条件被满足 90% 

## time complexity
>> o(logn)
## space complexity
>> o(1)

## 几种类型的双指针及相关题目


--- 
## template 

```python
class Solution:
    def binary_search(self, nums, target):

        # corner case 处理 - 这里等价于num is None or len(num) == 0
        if not nums: return -1 

        # 初始化
        start, end = 0, len(nums) - 1

        # 用start + 1 < end 而不是 start < end 的目的是为了避免死循环
        # the first position of target 的情况下不会出现死循环，但是在last position of target 的情况下会出现死循环
        # example, nums = [1, 1]， target = 1 

        while start + 1 < end:
            mid = start + (end - start) // 2 
            # or mid = (start + end) // 2

            # > = < 的逻辑先分开写，然后在看看 = 的情况是否能合并到其他分支里
            if nums[mid] < target: 
                start = mid 

            elif nums[mid] == target:
                end = mid 

            else: 
                end = mid 

            # 因为上面的循环退出条件是 start + 1 < end 
            # 因此这里循环结束的时候， start 和 end 的关系是相邻关系 (1和2，3和4 这种)
            # 因此需要再单独判断 start 和 end 这两个数谁是我们想要的答案
            # 如果是找 first position of target 就先看start， 否则就先看end 

            if nums[start] == target:
                return start 

            if nums[end] == target: 
                return end 

        return -1 




```






