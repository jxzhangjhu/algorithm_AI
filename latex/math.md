# Math 相关题型


### randomized 


### 进位制理解和操作 - digit 
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

400. Nth Digit  https://leetcode.com/problems/nth-digit/ 

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