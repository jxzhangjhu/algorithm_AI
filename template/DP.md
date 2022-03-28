

```python 
class Solution:
    def maximumScore(self, nums: List[int], multipliers: List[int]) -> int:
        # lru_cache from functools automatically memoizes the function
        @lru_cache(2000)
        def dp(i, left):
            # Base case
            if i == m:
                return 0

            mult = multipliers[i]
            right = n - 1 - (i - left)
            
            # Recurrence relation
            return max(mult * nums[left] + dp(i + 1, left + 1), 
                       mult * nums[right] + dp(i + 1, left))
                       
        n, m = len(nums), len(multipliers)
        return dp(0, 0)


class Solution:
    def maximumScore(self, nums: List[int], multipliers: List[int]) -> int:
        n, m = len(nums), len(multipliers)
        dp = [[0] * (m + 1) for _ in range(m + 1)]
        
        for i in range(m - 1, -1, -1):
            for left in range(i, -1, -1):
                mult = multipliers[i]
                right = n - 1 - (i - left)
                dp[i][left] = max(mult * nums[left] + dp[i + 1][left + 1], 
                                  mult * nums[right] + dp[i + 1][left])        
        return dp[0][0]

```































## 740. Delete and Earn 

```python
class Solution:
    def deleteAndEarn(self, nums: List[int]) -> int:
        points = defaultdict(int)
        max_number = 0
        # Precompute how many points we gain from taking an element
        for num in nums:
            points[num] += num
            max_number = max(max_number, num)

        @cache
        def max_points(num):
            # Check for base cases
            if num == 0:
                return 0
            if num == 1:
                return points[1]
            
            # Apply recurrence relation
            return max(max_points(num - 1), max_points(num - 2) + points[num])
        
        return max_points(max_number)


class Solution:
    def deleteAndEarn(self, nums: List[int]) -> int:
        points = defaultdict(int)
        max_number = 0
        # Precompute how many points we gain from taking an element
        for num in nums:
            points[num] += num
            max_number = max(max_number, num)

        # Declare our array along with base cases
        max_points = [0] * (max_number + 1)
        max_points[1] = points[1]

        for num in range(2, len(max_points)):
            # Apply recurrence relation
            max_points[num] = max(max_points[num - 1], max_points[num - 2] + points[num])
        
        return max_points[max_number]


class Solution:
    def deleteAndEarn(self, nums: List[int]) -> int:
        points = {}
        max_number = 0
        # Precompute how many points we gain from taking an element
        for num in nums:
            points[num] = points.get(num, 0) + num
            max_number = max(max_number, num)
        
        # Base cases
        two_back = 0
        one_back = points.get(1, 0)
        
        for num in range(2, max_number + 1):
            two_back, one_back = one_back, max(one_back, two_back + points.get(num, 0))

        return one_back



class Solution:
    def deleteAndEarn(self, nums: List[int]) -> int:
        points = defaultdict(int)
        # Precompute how many points we gain from taking an element
        for num in nums:
            points[num] += num
            
        elements = sorted(points.keys())
        two_back = 0
        one_back = points[elements[0]]
        
        for i in range(1, len(elements)):
            current_element = elements[i]
            if current_element == elements[i - 1] + 1:
                # The 2 elements are adjacent, cannot take both - apply normal recurrence
                two_back, one_back = one_back, max(one_back, two_back + points[current_element])
            else:
                # Otherwise, we don't need to worry about adjacent deletions
                two_back, one_back = one_back, one_back + points[current_element]

        return one_back


class Solution:
    def deleteAndEarn(self, nums: List[int]) -> int:
        points = defaultdict(int)
        max_number = 0
        for num in nums:
            points[num] += num
            max_number = max(max_number, num)
        
        two_back = one_back = 0
        n = len(points)
        if max_number < n + n * log(n, 2):
            one_back = points[1]
            for num in range(2, max_number + 1):
                two_back, one_back = one_back, max(one_back, two_back + points[num])
        else:
            elements = sorted(points.keys())
            one_back = points[elements[0]]     
            for i in range(1, len(elements)):
                current_element = elements[i]
                if current_element == elements[i - 1] + 1:
                    two_back, one_back = one_back, max(one_back, two_back + points[current_element])
                else:
                    two_back, one_back = one_back, one_back + points[current_element]

        return one_back

```