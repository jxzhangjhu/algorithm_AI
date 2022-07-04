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


### 1291. Sequential Digits https://leetcode.com/problems/sequential-digits/ 
An integer has sequential digits if and only if each digit in the number is one more than the previous digit.
Return a sorted list of all the integers in the range [low, high] inclusive that have sequential digits.

```
Example 1:

Input: low = 100, high = 300
Output: [123,234]
Example 2:
Input: low = 1000, high = 13000
Output: [1234,2345,3456,4567,5678,6789,12345]

Constraints:
10 <= low <= high <= 10^9
```

```python
class Solution:
    def sequentialDigits(self, low: int, high: int) -> List[int]:
        # time o(1), space o(1)
        # 这个想不到没法做，太恶心了， 主要需要知道上下限10 <= low <= high <= 10^9 这个信息很重要
        # 对sample这个取样
        # 用str(low) 操作上下限要容易很多，之前在想如何得到它的位数！
        sample = "123456789"
        n = 10
        nums = []
        for length in range(len(str(low)), len(str(high)) + 1):
            for start in range(n - length):
                num = int(sample[start: start + length])
                if num >= low and num <= high:
                    nums.append(num)
        return nums
```


### 66. Plus One https://leetcode.com/problems/plus-one/ 

You are given a large integer represented as an integer array digits, where each digits[i] is the ith digit of the integer. The digits are ordered from most significant to least significant in left-to-right order. The large integer does not contain any leading 0's.

Increment the large integer by one and return the resulting array of digits.

```
Example 1:

Input: digits = [1,2,3]
Output: [1,2,4]
Explanation: The array represents the integer 123.
Incrementing by one gives 123 + 1 = 124.
Thus, the result should be [1,2,4].
Example 2:

Input: digits = [4,3,2,1]
Output: [4,3,2,2]
Explanation: The array represents the integer 4321.
Incrementing by one gives 4321 + 1 = 4322.
Thus, the result should be [4,3,2,2].
Example 3:

Input: digits = [9]
Output: [1,0]
Explanation: The array represents the integer 9.
Incrementing by one gives 9 + 1 = 10.
Thus, the result should be [1,0].
```
> 从medium 到 easy，但其实不简单
```python
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
# Input: digits = [4,3,2,1]
# Output: [4,3,2,2]
# Explanation: The array represents the integer 4321.
# Incrementing by one gives 4321 + 1 = 4322.
# Thus, the result should be [4,3,2,2].
        # 写成函数好很多，要不太乱了, 没有什么具体的考点，就是simulate
        # time o(n), n is the length of number, space o(n)
        def num_list(num, length):
            res = []
            for i in range(0, length):
                digit = num // (10 ** (length - i - 1))
                num -= digit * (10 ** (length - i - 1))
                res.append(digit)
            return res
        
        n = len(digits)
        # n - 1 will be the real digits 
        j = n - 1 
        number = 0
        for i in range(n):
            number += digits[i] * (10**(j)) 
            j -= 1
            
        newnumber = number + 1   
        if newnumber // (10 ** n) == 1:
            return [1] + [0] * (n)
        else:
            return num_list(newnumber, n)
                
        
#         # 官方答案
        n = len(digits)
        # move along the input array starting from the end
        for i in range(n):
            idx = n - 1 - i
            # set all the nines at the end of array to zeros
            if digits[idx] == 9:
                digits[idx] = 0
            # here we have the rightmost not-nine
            else:
                # increase this rightmost not-nine by 1
                digits[idx] += 1
                # and the job is done
                return digits

        # we're here because all the digits are nines
        return [1] + digits
``` 

### 67. Add Binary https://leetcode.com/problems/add-binary/ [easy]
Given two binary strings a and b, return their sum as a binary string.
```
Example 1:

Input: a = "11", b = "1"
Output: "100"
Example 2:

Input: a = "1010", b = "1011"
Output: "10101"
``` 
> 二进制的问题，很多类似的math problem
```python
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        # brute force
        # return '{0:b}'.format(int(a, 2) + int(b, 2)) # 不理解，没用过， time o(N+M)的时间
    
    # 进位的方法
        res = ''
        carry = 0 
        i, j = len(a) - 1, len(b) - 1
        while i >= 0 or j >= 0 or carry != 0 :
            digitA = int(a[i]) if i >= 0 else 0 # 注意这个
            digitB = int(b[j]) if j >= 0 else 0
            sum = digitA + digitB + carry
            if sum >= 2: 
                carry = 1
                sum -= 2
            else:
                carry = 0 
            res += str(sum)
            i -= 1
            j -= 1
        return res[::-1]
``` 


### 415. Add Strings https://leetcode.com/problems/add-strings/ 
Given two non-negative integers, num1 and num2 represented as string, return the sum of num1 and num2 as a string.

You must solve the problem without using any built-in library for handling large integers (such as BigInteger). You must also not convert the inputs to integers directly.

```
Example 1:

Input: num1 = "11", num2 = "123"
Output: "134"
Example 2:

Input: num1 = "456", num2 = "77"
Output: "533"
Example 3:

Input: num1 = "0", num2 = "0"
Output: "0"
```
> string 操作2个数相加！
```python
class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        # 类似的几个题，都是进位操作
        res = []
        carry = 0
        p1 = len(num1) - 1
        p2 = len(num2) - 1
        while p1 >= 0 or p2 >= 0:
            x1 = ord(num1[p1]) - ord('0') if p1 >= 0 else 0
            x2 = ord(num2[p2]) - ord('0') if p2 >= 0 else 0
            value = (x1 + x2 + carry) % 10
            carry = (x1 + x2 + carry) // 10
            res.append(value)
            p1 -= 1
            p2 -= 1
        
        if carry:
            res.append(carry)
        
        return ''.join(str(x) for x in res[::-1])
```


### 989. Add to Array-Form of Integer https://leetcode.com/problems/add-to-array-form-of-integer/ 
The array-form of an integer num is an array representing its digits in left to right order. For example, for num = 1321, the array form is [1,3,2,1]. Given num, the array-form of an integer, and an integer k, return the array-form of the integer num + k.
```
Example 1:

Input: num = [1,2,0,0], k = 34
Output: [1,2,3,4]
Explanation: 1200 + 34 = 1234
Example 2:

Input: num = [2,7,4], k = 181
Output: [4,5,5]
Explanation: 274 + 181 = 455
Example 3:

Input: num = [2,1,5], k = 806
Output: [1,0,2,1]
Explanation: 215 + 806 = 1021
```
> array + num 操作，都是类似的题
```python
class Solution:
    def addToArrayForm(self, num: List[int], k: int) -> List[int]:
        # brute force, time o(n), space o(n) 
        n = len(num)
        new = 0 
        for i in range(n):
            new += num[i] * 10 ** (n - i - 1)
        newnum = new + k 
        res = []
        while newnum > 0:
            reminder = newnum % 10 
            newnum = newnum // 10
            res.append(reminder)
        return res[::-1]
```


### 43. Multiply Strings https://leetcode.com/problems/multiply-strings/ 
Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2, also represented as a string. Note: You must not use any built-in BigInteger library or convert the inputs to integer directly.
```
Example 1:

Input: num1 = "2", num2 = "3"
Output: "6"
Example 2:

Input: num1 = "123", num2 = "456"
Output: "56088"
``` 
> 这几个最难的一个题了！
```python
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        if num1 == "0" or num2 == "0":
            return "0"
        ans = "0"
        m, n = len(num1), len(num2)
        for i in range(n - 1, -1, -1):
            add = 0
            y = int(num2[i])
            curr = ["0"] * (n - i - 1)
            for j in range(m - 1, -1, -1):
                product = int(num1[j]) * y + add
                curr.append(str(product % 10))
                add = product // 10
            if add > 0:
                curr.append(str(add))
            curr = "".join(curr[::-1])
            ans = self.addStrings(ans, curr)
        return ans
    
    def addStrings(self, num1: str, num2: str) -> str:
        i, j = len(num1) - 1, len(num2) - 1
        add = 0
        ans = list()
        while i >= 0 or j >= 0 or add != 0:
            x = int(num1[i]) if i >= 0 else 0
            y = int(num2[j]) if j >= 0 else 0
            result = x + y + add
            ans.append(str(result % 10))
            add = result // 10
            i -= 1
            j -= 1
        return "".join(ans[::-1])
``` 

### 2. Add Two Numbers https://leetcode.com/problems/add-two-numbers/

You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.
```
Example 1:


Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]
Explanation: 342 + 465 = 807.
Example 2:

Input: l1 = [0], l2 = [0]
Output: [0]
Example 3:

Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
Output: [8,9,9,9,0,0,0,1]
```
> linked list 需要自己写list 

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        # 当前指针，结果链表
        result = curr = ListNode()
        # 进位项
        remainder = 0
        # 非空满足循环条件
        while l1 or l2 :
            x = l1.val if l1 else 0
            y = l2.val if l2 else 0
            total = x + y + remainder
            curr.next = ListNode(total%10)
            remainder = total//10
            # 🚩防止某一链表已经为空，空链表.next会报错
            if l1 : l1 = l1.next
            if l2 : l2 = l2.next
            curr = curr.next

        if remainder : curr.next = ListNode(remainder)
        return result.next
```

### 445. Add Two Numbers II https://leetcode.com/problems/add-two-numbers-ii/
You are given two non-empty linked lists representing two non-negative integers. The most significant digit comes first and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.
```
Example 1:

Input: l1 = [7,2,4,3], l2 = [5,6,4]
Output: [7,8,0,7]
Example 2:

Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [8,0,7]
Example 3:

Input: l1 = [0], l2 = [0]
Output: [0]
``` 
> linked list, 更难的
```python
# # Definition for singly-linked list.
# # class ListNode:
# #     def __init__(self, val=0, next=None):
# #         self.val = val
# #         self.next = next
# class Solution:
#     def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        last = None
        while head:
            # keep the next node
            tmp = head.next
            # reverse the link
            head.next = last
            # update the last node and the current node
            last = head
            head = tmp
        
        return last
    
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        # reverse lists
        l1 = self.reverseList(l1)
        l2 = self.reverseList(l2)
        
        head = None
        carry = 0
        while l1 or l2:
            # get the current values 
            x1 = l1.val if l1 else 0
            x2 = l2.val if l2 else 0
            
            # current sum and carry
            val = (carry + x1 + x2) % 10
            carry = (carry + x1 + x2) // 10
            
            # update the result: add to front
            curr = ListNode(val)
            curr.next = head
            head = curr
            
            # move to the next elements in the lists
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None

        if carry:
            curr = ListNode(carry)
            curr.next = head
            head = curr

        return head
``` 


### 371. Sum of Two Integers https://leetcode.com/problems/sum-of-two-integers/ 
Given two integers a and b, return the sum of the two integers without using the operators + and -.
```
Example 1:

Input: a = 1, b = 2
Output: 3
Example 2:

Input: a = 2, b = 3
Output: 5
```
> bit operation 位运算的题
```python
class Solution:
    def getSum(self, a: int, b: int) -> int:
        
        # bit operation 这种题没研究
        x, y = abs(a), abs(b)
        # ensure x >= y
        if x < y:
            return self.getSum(b, a)  
        sign = 1 if a > 0 else -1
        
        if a * b >= 0:
            # sum of two positive integers
            while y:
                x, y = x ^ y, (x & y) << 1
        else:
            # difference of two positive integers
            while y:
                x, y = x ^ y, ((~x) & y) << 1
        
        return x * sign
``` 

### 9. Palindrome Number https://leetcode.com/problems/palindrome-number/ 
Given an integer x, return true if x is palindrome integer.

An integer is a palindrome when it reads the same backward as forward.

For example, 121 is a palindrome while 123 is not.
 
```
Example 1:

Input: x = 121
Output: true
Explanation: 121 reads as 121 from left to right and from right to left.
Example 2:

Input: x = -121
Output: false
Explanation: From left to right, it reads -121. From right to left, it becomes 121-. Therefore it is not a palindrome.
Example 3:

Input: x = 10
Output: false
Explanation: Reads 01 from right to left. Therefore it is not a palindrome.

```
> 进位运算的题
```python
class Solution:
    def isPalindrome(self, x: int) -> bool:

        if x < 0 or (x > 0 and x%10 == 0):   # if x is negative, return False. if x is positive and last digit is 0, that also cannot form a palindrome, return False.
            return False

        result = 0
        while x > result:
            result = result * 10 + x % 10
            x = x // 10

        return True if (x == result or x == result // 10) else False
``` 


✅✅✅ 罗马数字转换 ✅✅✅ 

### 13. Roman to Integer

Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
For example, 2 is written as II in Roman numeral, just two ones added together. 12 is written as XII, which is simply X + II. The number 27 is written as XXVII, which is XX + V + II.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

I can be placed before V (5) and X (10) to make 4 and 9. 
X can be placed before L (50) and C (100) to make 40 and 90. 
C can be placed before D (500) and M (1000) to make 400 and 900.
Given a roman numeral, convert it to an integer.

```
Example 1:

Input: s = "III"
Output: 3
Explanation: III = 3.
Example 2:

Input: s = "LVIII"
Output: 58
Explanation: L = 50, V= 5, III = 3.
Example 3:

Input: s = "MCMXCIV"
Output: 1994
Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.
```
> 不知道要考什么？hahstable + math 
```python
values = {
    "I": 1,
    "V": 5,
    "X": 10,
    "L": 50,
    "C": 100,
    "D": 500,
    "M": 1000,
}

# 比较麻烦，不知道要考什么？
class Solution:
    def romanToInt(self, s: str) -> int:
        total = 0
        i = 0
        while i < len(s):
            # If this is the subtractive case.
            if i + 1 < len(s) and values[s[i]] < values[s[i + 1]]:
                total += values[s[i + 1]] - values[s[i]]
                i += 2
            # Else this is NOT the subtractive case.
            else:
                total += values[s[i]]
                i += 1
        return total
    
# time o(1), space o(1)
```

### 12. Integer to Roman

Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
For example, 2 is written as II in Roman numeral, just two one's added together. 12 is written as XII, which is simply X + II. The number 27 is written as XXVII, which is XX + V + II.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

I can be placed before V (5) and X (10) to make 4 and 9. 
X can be placed before L (50) and C (100) to make 40 and 90. 
C can be placed before D (500) and M (1000) to make 400 and 900.
Given an integer, convert it to a roman numeral.

```
Example 1:

Input: num = 3
Output: "III"
Explanation: 3 is represented as 3 ones.
Example 2:

Input: num = 58
Output: "LVIII"
Explanation: L = 50, V = 5, III = 3.
Example 3:

Input: num = 1994
Output: "MCMXCIV"
Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.
```

```python
class Solution:
    def intToRoman(self, num: int) -> str:
        thousands = ["", "M", "MM", "MMM"]
        hundreds = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"]
        tens = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"]
        ones = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
        return (thousands[num // 1000] + hundreds[num % 1000 // 100] 
               + tens[num % 100 // 10] + ones[num % 10])
``` 
