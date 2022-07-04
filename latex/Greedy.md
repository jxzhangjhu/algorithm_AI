# Greedy

## 贪心策略是最佳的，没有特别好的办法，或者直接模拟

### 2214. Minimum Health to Beat Game https://leetcode.com/problems/minimum-health-to-beat-game/ 
You are playing a game that has n levels numbered from 0 to n - 1. You are given a 0-indexed integer array damage where damage[i] is the amount of health you will lose to complete the ith level.

You are also given an integer armor. You may use your armor ability at most once during the game on any level which will protect you from at most armor damage.

You must complete the levels in order and your health must be greater than 0 at all times to beat the game.

Return the minimum health you need to start with to beat the game.

```
Example 1:

Input: damage = [2,7,4,3], armor = 4
Output: 13
Explanation: One optimal way to beat the game starting at 13 health is:
On round 1, take 2 damage. You have 13 - 2 = 11 health.
On round 2, take 7 damage. You have 11 - 7 = 4 health.
On round 3, use your armor to protect you from 4 damage. You have 4 - 0 = 4 health.
On round 4, take 3 damage. You have 4 - 3 = 1 health.
Note that 13 is the minimum health you need to start with to beat the game.
Example 2:

Input: damage = [2,5,3,4], armor = 7
Output: 10
Explanation: One optimal way to beat the game starting at 10 health is:
On round 1, take 2 damage. You have 10 - 2 = 8 health.
On round 2, use your armor to protect you from 5 damage. You have 8 - 0 = 8 health.
On round 3, take 3 damage. You have 8 - 3 = 5 health.
On round 4, take 4 damage. You have 5 - 4 = 1 health.
Note that 10 is the minimum health you need to start with to beat the game.
Example 3:

Input: damage = [3,3,3], armor = 0
Output: 10
Explanation: One optimal way to beat the game starting at 10 health is:
On round 1, take 3 damage. You have 10 - 3 = 7 health.
On round 2, take 3 damage. You have 7 - 3 = 4 health.
On round 3, take 3 damage. You have 4 - 3 = 1 health.
Note that you did not use your armor ability.
```

```python
class Solution:
    def minimumHealth(self, damage: List[int], armor: int) -> int:
        # 贪心
        if max(damage) >= armor:
            return sum(damage) - armor  + 1 
        else:
            return sum(damage) - max(damage) + 1
``` 


### 1921. Eliminate Maximum Number of Monsters  https://leetcode.com/problems/eliminate-maximum-number-of-monsters/ 
You are playing a video game where you are defending your city from a group of n monsters. You are given a 0-indexed integer array dist of size n, where dist[i] is the initial distance in kilometers of the ith monster from the city.

The monsters walk toward the city at a constant speed. The speed of each monster is given to you in an integer array speed of size n, where speed[i] is the speed of the ith monster in kilometers per minute.

You have a weapon that, once fully charged, can eliminate a single monster. However, the weapon takes one minute to charge.The weapon is fully charged at the very start.

You lose when any monster reaches your city. If a monster reaches the city at the exact moment the weapon is fully charged, it counts as a loss, and the game ends before you can use your weapon.

Return the maximum number of monsters that you can eliminate before you lose, or n if you can eliminate all the monsters before they reach the city.

```
Example 1:

Input: dist = [1,3,4], speed = [1,1,1]
Output: 3
Explanation:
In the beginning, the distances of the monsters are [1,3,4]. You eliminate the first monster.
After a minute, the distances of the monsters are [X,2,3]. You eliminate the second monster.
After a minute, the distances of the monsters are [X,X,2]. You eliminate the thrid monster.
All 3 monsters can be eliminated.
Example 2:

Input: dist = [1,1,2,3], speed = [1,1,1,1]
Output: 1
Explanation:
In the beginning, the distances of the monsters are [1,1,2,3]. You eliminate the first monster.
After a minute, the distances of the monsters are [X,0,1,2], so you lose.
You can only eliminate 1 monster.
Example 3:

Input: dist = [3,2,4], speed = [5,3,2]
Output: 1
Explanation:
In the beginning, the distances of the monsters are [3,2,4]. You eliminate the first monster.
After a minute, the distances of the monsters are [X,0,2], so you lose.
You can only eliminate 1 monster.
``` 
> 纯贪心 + 排序 + 模拟，没有什么难度和技巧，之前的写法直接暴力超时了！
```python
class Solution:
    def eliminateMaximum(self, dist: List[int], speed: List[int]) -> int:
#         res = 1
#         temp = 1
#         diff_min = +inf
#         while temp:
#             for i in range(1, len(dist)):
#                 diff = dist[i] - speed[i]
#                 diff_min = min(diff_min, diff)
#             if diff_min >= 1:
#                 res += 1
#             else:
#                 temp = 0 
        
#         return res
        
    # 排序 + 贪心， time o(nlogn), space o(n)
        n = len(dist)
        # 每个怪物的最晚可被消灭时间
        time = [(dist[i] - 1) // speed[i] for i in range(n)]
        time.sort()
        for i in range(n):
            if time[i] < i:
                # 无法消灭该怪物，返回消灭的怪物数量
                return i
        # 成功消灭全部怪物
        return n

# 作者：LeetCode-Solution
# 链接：https://leetcode.cn/problems/eliminate-maximum-number-of-monsters/solution/xiao-mie-guai-wu-de-zui-da-shu-liang-by-0ou2p/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
``` 



### 605. Can Place Flowers https://leetcode.com/problems/can-place-flowers/ 
You have a long flowerbed in which some of the plots are planted, and some are not. However, flowers cannot be planted in adjacent plots.

Given an integer array flowerbed containing 0's and 1's, where 0 means empty and 1 means not empty, and an integer n, return if n new flowers can be planted in the flowerbed without violating the no-adjacent-flowers rule.

```
Example 1:

Input: flowerbed = [1,0,0,0,1], n = 1
Output: true
Example 2:

Input: flowerbed = [1,0,0,0,1], n = 2
Output: false
``` 
> greedy 策略

```python
class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        # greey 连续3个0 就能种花, 贪心策略
        count = 0 
        flowerbed = [0] + flowerbed + [0]     
        for i in range(1,len(flowerbed)-1):
            if flowerbed[i] ==0 and flowerbed[i-1] !=1 and flowerbed[i+1] !=1:
                count+=1
                flowerbed[i] =1 # 要把中间的中上！
        return count>=n
```