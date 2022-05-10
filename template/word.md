## Word 题

- #127. Word Ladder https://leetcode.com/problems/word-ladder/ [hard]
```python
#         # 这个题比较复杂，几个点要想到
#         # 1. 如何处理转换，也就是string要熟，用26个字母的方式
#         # 2. 要想到是BFS
#           3. 时间复杂度 26xLxL, L 是单词长度， 这个比较麻烦分析！
        wordSet = set(wordList)
        queue = deque()
        queue.append(beginWord)
        visited = set()
        visited.add(beginWord)
        
        level = 0
        while queue:
            # same logic as # 102 Binary Tree Level order Traversal
            queueLen = len(queue)
            
            # rest is similar to # 102 as well, if child exist, then add to queue
            for _ in range(queueLen):
                curr = queue.popleft() # 直接pop出来就是单词
                if curr == endWord:
                    return level + 1
                
                print(curr)
                tmpWord = list(curr) # 转换为list来考虑
                print(tmpWord)
                # 处理是否下一个word在wordset， 就是一个暴力搜索 + 记忆判断
                for i in range(len(tmpWord)):
                    for j in range(26):
                        tmpWord[i] = chr(ord('a') + j) # 这个就不熟悉
                        formedWord = "".join(tmpWord) # list to string 
                        if formedWord in wordSet and formedWord not in visited:
                            queue.append(formedWord) # 基于现在pop出来的头单词，然后有哪些可能加到队尾！
                            visited.add(formedWord) # 加这个就是记录咩有看到的新单词
                    
                    tmpWord = list(curr)
        
            level += 1
        return 0
```

- #126. Word Ladder II https://leetcode.com/problems/word-ladder-ii/ [hard]
    > 需要记录路径！
```python
class Solution:
    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        
        # 这个比127难，因为需要返回path，也就是说要用backtracking，如何记录这些东西
        wordList.append(beginWord)
        nei = collections.defaultdict(list)

        #Creating adjs  list using pattern over the words
        for w in wordList :
            for k in range(len(w)) :
                pattern = w[:k] + '-' + w[k + 1:]
                nei[pattern].append(w)

        q = collections.deque([(beginWord, [beginWord])])
        vis = set([beginWord])
        res = []

        while q :
            auxSet = set() #this enable have control over the graph paths. The key for this problem

            for s in range(len(q)) :
                w, seq = q.popleft()
                if w == endWord :
                    res.append(seq)      

                for k in range(len(w)) : #sech adjs list 
                    pattern = w[:k] + '-' + w[k + 1:]
                    for adj in nei[pattern] :
                        if adj not in vis :                        
                            auxSet.add(adj)
                            q.append((adj, seq[:]+[adj]))

            vis.update(auxSet) 

        return res
```