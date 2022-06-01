# Prefix sum 

## 很多是subarray的题

### 238. Product of Array Except Self https://leetcode.com/problems/product-of-array-except-self/ 

Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i]. The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer. You must write an algorithm that runs in O(n) time and without using the division operation.
```
Example 1:

Input: nums = [1,2,3,4]
Output: [24,12,8,6]
Example 2:

Input: nums = [-1,1,0,-3,3]
Output: [0,0,9,0,0]
```

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:        
        n = len(nums)
        ans = [0]*n
        
        ans[0] = 1
        for i in range(1, n):
            ans[i] = ans[i-1]*nums[i-1]

        R = 1
        for i in reversed(range(n)):
            ans[i] = ans[i] * R
            R *= nums[i]

        return ans
``` 


### 325. Maximum Size Subarray Sum Equals k  https://leetcode.com/problems/maximum-size-subarray-sum-equals-k/ 
Given an integer array nums and an integer k, return the maximum length of a subarray that sums to k. If there is not one, return 0 instead.
```
Example 1:

Input: nums = [1,-1,5,-2,3], k = 3
Output: 4
Explanation: The subarray [1, -1, 5, -2] sums to 3 and is the longest.
Example 2:

Input: nums = [-2,-1,2,1], k = 1
Output: 2
Explanation: The subarray [-1, 2] sums to 1 and is the longest.
```

```python
class Solution:
    def maxSubArrayLen(self, nums: List[int], k: int) -> int:
        # presum的题
        hashmap = {0:-1}
        pre_sum = 0
        res = 0
        for i, num in enumerate(nums):
            pre_sum += num
            # Check if all of the numbers seen so far sum to k. 非常有必要！否则容易错！
            if pre_sum == k:
                res = i + 1
            # If any subarray seen so far sums to k, then
            # update the length of the longest_subarray. 
            if pre_sum - k in hashmap:
                res = max(res, i - hashmap[pre_sum - k])
            # Only add the current prefix_sum index pair to the 
            # map if the prefix_sum is not already in the map.
            if pre_sum not in hashmap: # 这个判断有必要，因为可能正负都有，之前的题没有判断这个因为递增
                hashmap[pre_sum] = hashmap.get(pre_sum, 0) + i
        
        return res 
            
    # time o(n), space o(n)
```

### 560. Subarray Sum Equals K
Given an array of integers nums and an integer k, return the total number of subarrays whose sum equals to k. A subarray is a contiguous non-empty sequence of elements within an array.
```
Example 1:

Input: nums = [1,1,1], k = 2
Output: 2
Example 2:

Input: nums = [1,2,3], k = 3
Output: 2
```

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        
# https://leetcode-cn.com/problems/subarray-sum-equals-k/solution/qian-zhui-he-si-xiang-560-he-wei-kde-zi-shu-zu-by-/
    
        # 很高频的，前缀和的几种写法
        # # way1 - brute force 超时，o(n^2), space o(1)
        # res = 0
        # for i in range(len(nums)):
        #     for j in range(i, len(nums)):
        #         if sum(nums[i:j+1]) == k:
        #             res += 1
        # return res
        
        
        # prefix sum - 也是超时了  o(n^2), space o (n) 因为有pre的空间      
#         cnt, n =  0, len(nums)
#         pre = [0] * (n + 1)
#         for i in range(1, n + 1):
#             pre[i] = pre[i - 1] + nums[i - 1]
            
#         for i in range(1, n + 1):
#             for j in range(i, n + 1):
#                 if (pre[j] - pre[i - 1] == k): cnt += 1
#         return cnt

        # hashmap + pre_sum 这个写法更适合我之前的习惯
        pre, res = 0, 0
        count = dict()
        for num in nums:
            pre += num
            if pre == k: res += 1
            res += count.get(pre-k, 0)
            count[pre] = count.get(pre, 0) + 1 # 这个写法更好一些，和sliding window一样

        return res


#         pre_sum = collections.defaultdict(int) 
#         res, cur_pre_sum = 0, 0
#         for i in range(len(nums)):
#             cur_pre_sum += nums[i]
#             if cur_pre_sum - k in pre_sum:
#                 res += pre_sum[cur_pre_sum]
#             pre_sum[cur_pre_sum] += 1
#         return res
        

        # # 记录 绿色 "前缀和" (从0到i的前缀和) 的 值和出现的次数.
        # pre_sum = collections.defaultdict(int) 
        # # 初始化 前缀和 为 0 的 子序列 出现了 一次. 
        # # 对应 第一类情况, 上面的 if cur_pre_sum - k == 0 语句
        # pre_sum[0] = 1  
        # # 记录 当前 位置的 前缀和
        # cur_pre_sum = 0
        # # 用于记录结果
        # res = 0
        # for i in range(len(nums)):
        #     cur_pre_sum += nums[i]  # 计算 当前位置的 前缀和
        #     # cur_sum - k 是我们想找的前缀和 nums[0..j]
        #     # 如果前面有这个前缀和, 则直接更新答案
        #     green_sum = cur_pre_sum - k
        #     if green_sum in pre_sum:
        #         res += pre_sum[green_sum]
        #     # 每次计算都将前缀和加入字典
        #     pre_sum[cur_pre_sum] += 1
        # return res
```

### 974. Subarray Sums Divisible by K 
Given an integer array nums and an integer k, return the number of non-empty subarrays that have a sum divisible by k. A subarray is a contiguous part of an array.
```
Example 1:

Input: nums = [4,5,0,-2,-3,1], k = 5
Output: 7
Explanation: There are 7 subarrays with a sum divisible by k = 5:
[4, 5, 0, -2, -3, 1], [5], [5, 0], [5, 0, -2, -3], [0], [0, -2, -3], [-2, -3]
Example 2:

Input: nums = [5], k = 9
Output: 0
```
```python
class Solution:
    def subarraysDivByK(self, nums: List[int], k: int) -> int:
        # res, cnt = 0, Counter({0:1})    # 定义哨兵节点，取余结果为0时，默认已经出现一次
        res, cnt = 0, {0:1}    # 改成{} 不能直接调用，会报错，如果之前没有出现
        pre = list(accumulate(nums, add))    # 计算前缀和数组

        for i in range(len(pre)):    # 遍历每个前缀和元素
            mod = (pre[i]+k) % k    # 因为可能存在负数，所以加上一个k，再计算对k取余结果
            res += cnt[mod]    # 加上哈希表中存储的mod对应的次数，更新可行方案数， 会报错
            cnt[mod] += 1    # 更新哈希表
            print('mod',mod)
            print('cnt',cnt)
            print('res',res)
            
        return res

#         # 这个题挺好的，不需要做判断，还是同余的道理，
#         pre_sum, res = 0, 0
#         hashmap = {} # 这样初始化，然后把prefix sum 求和的情况在for loop里面判断
        
#         for i, num in enumerate(nums):
#             pre_sum += num
#             if pre_sum % k == 0:
#                 res += 1
#             reminder = pre_sum % k            
#             res += hashmap.get(reminder,0)
#             hashmap[reminder] = hashmap.get(reminder, 0) + 1
            
#             #很多时候会考虑到判断，但是少一个解？
#             # if reminder in hashmap:
#             #     res += hashmap[reminder]
#             # else:
#             #     hashmap[reminder] = hashmap.get(reminder, 0) + 1
                
#         return res
```

### 1590. Make Sum Divisible by P 
Given an array of positive integers nums, remove the smallest subarray (possibly empty) such that the sum of the remaining elements is divisible by p. It is not allowed to remove the whole array. Return the length of the smallest subarray that you need to remove, or -1 if it's impossible. A subarray is defined as a contiguous block of elements in the array.
```
Example 1:

Input: nums = [3,1,4,2], p = 6
Output: 1
Explanation: The sum of the elements in nums is 10, which is not divisible by 6. We can remove the subarray [4], and the sum of the remaining elements is 6, which is divisible by 6.
Example 2:

Input: nums = [6,3,5,2], p = 9
Output: 2
Explanation: We cannot remove a single element to get a sum divisible by 9. The best way is to remove the subarray [5,2], leaving us with [6,3] with sum 9.
Example 3:

Input: nums = [1,2,3], p = 3
Output: 0
Explanation: Here the sum is 6. which is already divisible by 3. Thus we do not need to remove anything.
```

```python
class Solution:
    def minSubarray(self, nums: List[int], p: int) -> int:

        pre = list(accumulate(nums, add))
        mod = pre[-1] % p
        hashT = {0:-1}
        if mod == 0: return 0

        res = len(nums)
        for i in range(len(nums)):
            curmod = pre[i] % p
            tarmod = (curmod - mod + p) % p
            if tarmod in hashT:
                dis = i - hashT[tarmod]
                res = dis if dis < res else res
                if res == 1 and len(nums) != 1:
                    return 1
            hashT[curmod] = i
        if res == len(nums):
            res = -1
        return res
``` 