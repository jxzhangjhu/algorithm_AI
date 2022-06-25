# Math 相关题型


## Randomized 随机操作

这个题以及相关题目有很多可以考的
1. 2个uniform distribution 求和不是uniform，因为有很多重复的可能性，比如rand2() + rand2() 有多种可能性得到3， 这个问题也可以用数学公式推导，他的分布是三角分布
2. rand_x() 生成[1,x] 那么(rand_x - 1) * Y + rand_y() 可以生成rand_{xy} [1, xy]的随机数 - 这个定理很重要
3. rejected sampling， 生成之后可以用mod 操作来拒绝掉不想要的部分，但注意效率，可以迭代几次来提高效率
4. 扔骰子问题，或者抛硬币 https://leetcode.cn/problems/implement-rand10-using-rand7/solution/wei-rao-li-lun-yi-ge-bu-jun-yun-ying-bi-fo4ei/ 
5. 只有2个Gaussian求和是Gaussian. 2个Uniform的加减乘除都不是uniform 
6. 2个均匀分布 求和是三角分布，这个知乎讲得挺好的！  https://www.zhihu.com/question/27060339 


### 470. Implement Rand10() Using Rand7() https://leetcode.com/problems/implement-rand10-using-rand7/ 

Given the API rand7() that generates a uniform random integer in the range [1, 7], write a function rand10() that generates a uniform random integer in the range [1, 10]. You can only call the API rand7(), and you shouldn't call any other API. Please do not use a language's built-in random API.

Each test case will have one internal argument n, the number of times that your implemented function rand10() will be called while testing. Note that this is not an argument passed to rand10().

Example 1:
```
Input: n = 1
Output: [2]
```
Example 2:
```
Input: n = 2
Output: [2,8]
```
Example 3:
```
Input: n = 3
Output: [3,8,10]
```
Constraints:1 <= n <= 105
Follow up: What is the expected value for the number of calls to rand7() function? Could you minimize the number of calls to rand7()?


```python
class Solution:
    def rand10(self):
        """
        :rtype: int
        """
        while True:
            base = rand7() - 1
            new = (rand7() - 1) * 7 + rand7() # 49 
            if new <= 40:
                return new % 10 + 1
            
            
class Solution: # 解决follow-up的question！
    def rand10(self) -> int:
        while True:
            a = rand7()
            b = rand7()
            idx = (a - 1) * 7 + b
            if idx <= 40:
                return 1 + (idx - 1) % 10 # 拒绝了41-49 9个数
            a = idx - 40
            b = rand7()
            # get uniform dist from 1 - 63 - 拒绝61,62,63 三个数
            idx = (a - 1) * 7 + b
            if idx <= 60:
                return 1 + (idx - 1) % 10
            a = idx - 60
            b = rand7()
            # get uniform dist from 1 - 21 - 这样每次只拒绝21一个数
            idx = (a - 1) * 7 + b
            if idx <= 20:
                return 1 + (idx - 1) % 10
```

✅✅✅ 进位值相关题目 ✅✅✅ 

## 进位制理解和操作 - digit 
基本操作不熟悉！
```python
# 就是对进位不了解或者不熟悉
num = 123

print(num // 10)  # 最高位 + 次高位
print(num // 100) # 最高位
print(num // 1000) # 0

print(num % 10) # 个位
print(num % 100) # 十位 + 个位
print(num % 1000) # 百分位 + 十位 + 个位

# output
# 12
# 1
# 0
# 3
# 23
# 123
````

### 400. Nth Digit  https://leetcode.com/problems/nth-digit/ 

Given an integer n, return the nth digit of the infinite integer sequence [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...].

Example 1:
```
Input: n = 3
Output: 3
```
Example 2:
```
Input: n = 11
Output: 0
Explanation: The 11th digit of the sequence 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ... is a 0, which is part of the number 10.
```

``` python
class Solution:
    def totalDigits(self, length: int) -> int:
        digits = 0
        curCount = 9
        for curLength in range(1, length + 1):
            digits += curLength * curCount
            curCount *= 10
        return digits

    def findNthDigit(self, n: int) -> int:
        low, high = 1, 9
        while low < high:
            mid = (low + high) // 2
            if self.totalDigits(mid) < n:
                low = mid + 1
            else:
                high = mid
        d = low # 返回属于的位数
        
        prevDigits = self.totalDigits(d - 1) # 之前的所有数 digit求和
        index = n - prevDigits - 1 # 当前位数的index，比如2位数中排序
        start = 10 ** (d - 1) # 之前有多少个数，绝对number，比,1，2，3，...，100
        num = start + index // d   # n对应的当前的数
        digitIndex = index % d # 判断是d位数中第几个，百分位还是十分位，个位
        print(d, prevDigits,index,start, num, digitIndex)
        print(num // 10 ** (d - digitIndex - 1)) # 这是关键，返回一个digit，都要%10
        return num // 10 ** (d - digitIndex - 1) % 10

        # example 121
        # print 2 9 111 10 65 1
        # print 65
 
class Solution:
    def findNthDigit(self, n: int) -> int:
        cur, base = 1, 9
        while n > cur * base:
            n -= cur * base
            cur += 1
            base *= 10
        print(cur,n)
        n -= 1
        # 数字
        num = 10 ** (cur - 1) + n // cur
        print(num)
        # 数字里的第几位
        idx = n % cur
        print(idx)
        return num // (10 ** (cur - 1 - idx)) % 10
```


### 539. Minimum Time Difference https://leetcode.com/problems/minimum-time-difference/
Given a list of 24-hour clock time points in "HH:MM" format, return the minimum minutes difference between any two time-points in the list.
```
Example 1:

Input: timePoints = ["23:59","00:00"]
Output: 1
Example 2:

Input: timePoints = ["00:00","23:59","00:00"]
Output: 0
```
> 第一步的函数一定要熟悉，ord的操作！ 后面的一次遍历比较straightforward在排序的基础上，开始也想到了这个idea 

```python
# time - nlogn 因为有排序，如果加了优化，可能会小一些； space o(logn) or o(n)? 为排序需要的空间，取决于具体语言的实现。 

def getMinutes(t: str) -> int:
    # 计算总时间，min为单位， 1440 是最大值
    return ((ord(t[0]) - ord('0')) * 10 + ord(t[1]) - ord('0')) * 60 + (ord(t[3]) - ord('0')) * 10 + ord(t[4]) - ord('0')

class Solution:
    def findMinDifference(self, timePoints: List[str]) -> int:
        # 这个判断是如果大于1440，那么必然有相同的时间，鸽巢原理， 优化一下
        n = len(timePoints)
        if n > 1440:
            return 0
        
        timePoints.sort() # nlogn 无法减少
        ans = float('inf')
        t0Minutes = getMinutes(timePoints[0])
        preMinutes = t0Minutes
        for i in range(1, len(timePoints)):
            minutes = getMinutes(timePoints[i])
            ans = min(ans, minutes - preMinutes)  # 相邻时间的时间差
            preMinutes = minutes
        ans = min(ans, t0Minutes + 1440 - preMinutes)  # 首尾时间的时间差
        return ans
``` 

