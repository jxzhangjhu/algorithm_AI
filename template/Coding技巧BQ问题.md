# Coding interview 技巧

Qiao Li shared tips 

```
Tips of the interivew:
 
Technical Questions:
 
For the technical questions, they will focus on your technical breadth & depth of knowledge, problem solving, and coding.
Depth: They’ll want details about methods you’ve used in your projects and why, details about projects to see if they were creative and pragmatic, details about whether you’ve dived deep to check outliers/edge cases, details about how you evaluated your models, and details about the fundamentals in your field of expertise.
Breadth: They’ll want to know in detail about other techniques and methodologies you’ve used that are outside your area of expertise. 
Application/Problem Solving: The interviewer will ask an ambiguous problem solving question (case study). They’ll want to assess if you can productively discuss a problem which is slightly underspecified. They’ll want to see that you know to ask clarification questions, reason about the missing pieces of information, and progress towards a solution or model to the problem.

Coding: For Applied Scientists, they’ll assess your ability to meet SDE Junior level coding write code in any language of your choice, your ability to think algorithmically, analyze algorithm performance (runtime + space complexity and tradeoffs.), know general data structures (e.g., trees, graphs), take hints and discuss code with a SDE. Coding at the level of being able to manipulate data understanding basic data structures (e.g., Hashes, Lists, Arrays), Standard general algorithms (e.g., searching, sorting). If you can write the code in one production language (Python, Java, C++, etc.) and broad knowledge of coding methods (e.g., recursion)
Videos: In this section you’ll find videos of Amazonians describing the coding questions and ideal solutions/approaches in our on-site interviews.
Amazon Coding Question Tips (Hyperlinked)
Amazon Career Day (HackerRank) (Hyperlinked) Minute Marks 2:10-27:15
Two practice website: https://leetcode.com/ https://www.hackerrank.com/                
 
Tips for the Technical Questions:
 
- MOST IMPORTANT: The interviewer wants to see that you can properly gather the right requirements and can break down the problem. They will ask these questions in a very vague or ambiguous way. Never make assumptions. You’ll want to make sure to ask good clarifying questions before you start to work on the technical questions that are asked.
- ALWAYS try to think out loud as you solve the problem. This helps the interviewer understand your thought process. It’s also helpful in the case you steer away from the solution, they will give you hints to get back on track.
- ALWAYS stop and check your work every few minutes.
- Keep in mind that a problem can often be filed in several ways. Interviewers are often looking for whether the candidate chooses the simplest solution versus the complicated one. Choose the simplest one! But also, talk to the interviewer about the various possible solutions and why you chose the one you choose.
```



Amazon 官方tips 
https://www.youtube.com/watch?v=mjZpZ_wcYFg

1. CS 基础，data structure and algorithms
2. 启发式的解决问题，而不是random的解决
3. clean, logical maintainable code and people is easy to understand 

key principle 
1. 在没有看space之前，直接jump into的problem 立即 - 需要分析space？
2. try and disambiguate the problem 使问题具体化，通过问问题，理解input and output， 有哪些edge cases 我们可以考虑
3. talk this out loud, interacte not recitation 通过讲述，让别人理解what you are thinking and why and how you are trying to solve this problem


```example
1. 要写注释
2. implementation
3. testing  - edge cases， then optimization 考虑time， and space complexity 
4. 优化的空间

```





















# Amazon BQ核心考点，经验和优秀答案

- 亚麻官方提供的视频 sample  https://www.youtube.com/watch?v=CpcxVE5JIX4&t=138s 
1. STAR method, situation, task, action and result
2. need to quantify the impace with data! 量化的结果，metrics，提高了多少，数字！
2. need many details and future do not have the similar issue
3. not same example， 多准备一些例子
4. I not we，强调我的贡献
5. 解释重要性，强调其中的method和techniques 
6. 如果讲faliure, 要加上learning 你学到了什么？

- 来offer 亚麻BQ面试攻略: 是什么? 怎么答? 如何练习? 优秀答案长什么样? 
https://laioffer.medium.com/%E4%BA%9A%E9%BA%BBbq%E9%9D%A2%E8%AF%95%E6%94%BB%E7%95%A5-%E6%98%AF%E4%BB%80%E4%B9%88-%E6%80%8E%E4%B9%88%E7%AD%94-%E5%A6%82%E4%BD%95%E7%BB%83%E4%B9%A0-%E4%BC%98%E7%A7%80%E7%AD%94%E6%A1%88%E9%95%BF%E4%BB%80%E4%B9%88%E6%A0%B7-e5af142d4a8 

1. Situatiuon 情况阐述 - 当时遇到什么情况，点题！
2. Task 任务阐述 - 我的具体任务是什么，工作是什么， 点题！
3. Action 行动措施 - 采取了哪些行动和措施去解决问题 
4. Result 结果描述 - 得到了什么结果，完成与否，有何影响，都要是正面积极地，以后可以借鉴的，不会再有类似情况发生的


## 高频必考 - General BQ 
- introduce yourself 
- Why Amazon? Why this team and this position? 
- Why choose industry rather than academia? 
- What's your strength and weakness? 

## 22 Questions

### 1. Tell me about a time when you received negative feedback from your manager. How did you respond? [Earn Trust]
>你曾收到过来自同事或领导的负面反馈吗？你是如何改进的？ - 个人都有犯错的时候，收到负面反馈并不是难以启齿的事情, 把为什么收到负面反馈的原因说清，并且总结从中学到的经验、重新得到他人的认可才是重点。

- Stroy: LDRD project 开始和markus那个，缺少合作，和domain knowledge 最终实现好的效果
- S/T : 来ORNL开始的时候，项目不太了解，ML for physics，review 时间比较紧，我开始独立做，但是缺少background，review结果不是很理解，其实我应该早点寻求LDRD project manager的帮助， 说我应该早点寻求帮助
- A: 我意识到这个问题， 合理的评估自己的困难，找寻合适的domain expert，他来帮助我生成data，并且理解data，并且和project manager交流进度, domain knowledge is important 
- R: 我学会了自我评估和及时沟通，最终这个project 效果很多，我们发表了2篇论文，program manager很满意，给与了肯定，

### 2. Your team/co-workder/collabroator does not perform well, have bad impact to your work, how to handle that? [Ownership]
>如果你的小组同事经常表现不好，甚至影响到你工作，您是如何处理的？ - 因为团队是需要合作的。找到队友犯错的原因，帮助解决问题，完成工作任务才是重点。

- Story: LDRD project, Anomly detection with MDF, 明确他们的困难，还有credit 分配的问题，实现共赢！
- S/T: 他们不来meeting，比较缓慢，积极性不高
- A: 首先沟通了解困难和问题，发现他们是时间还有credit，他们想从应用方面体现，时间上和manager沟通调整我们的timeline，然后credit，我们从AI和计算方面证明，他们从manufacturing证明
- R: 效果很不错 - 我们的paper发表在neurips，已经一个dataset， 他们成功上线我们的方法，节约了大量人工标注的成本，提高了工作效率，5-10% performance 提升
- L: 学习到了沟通的重要性，同时协同能力，如何实现win-win的principle



### 4. **Failure/mistake** Tell me about a time you failed/the biggest mistake you made 
> 最大的失误和失败

- Story: 和mike准备ORN海军答辩的事
- S/T: 准备答辩，周末，时间很紧张，我们要研究Bayesian inference，small data 的问题，我因为控制random seed 导致data变化， 结果不对，海军的data
- A: 我仔细了check了code，并且和导师讨论了多次，然后重新跑所有的case，重新弄slides
- R: 我们的结果非常好，尽管时间很晚，我们工作到晚上10点，但得到了program manager的认可，后续，我导师因为这个工作获得了YIR from Naval，support的PHD study 
- L: 需要清楚原理，仔细，并且多和custom 沟通，通常我们认为没有问题，但这里是small data，我们需要真实的情况，海军需要根据他们的data来做！ 

### 5. **Challenge** - Most challenging project [highest standards, think big]
> 如何走出comfort zone，如何提升自己

- Story: inverse design problem 用normalizing flows这个
- S/T: why challenging 因为data比较少，DFT计算很贵，需要理解目标，之前都是forward problem，现在是inverse problem，如何利用AI ML去解决，这个方面的很少，我也缺乏这方面的经验
- A: 和同事manager多讨论，理解他们的需求； 主动学习data 生成和metric定义；implement invertible neural networks, 后续optimization
- R: 收到postive feedback from program manager，发表了nature，有github，highlights ornl
- L: 主动学习，克服挑战，突破自己的comfort zone很重要，不会说 can not do that, 而是let's try first and I can do better 


## 6. **Deadline** 

### Missing deadline [customer obsession, deliver results]
> 要突出如何让影响最小，让customer不受影响，为何会miss，最后结果是好的, 影响不是很大

- Story: 用DGS做一个demo，topology optimization for MDF，manager 突然说customer 提前1周时间去看这个demo，没办法搞定，需要超算
- S/T: deadline 提前，无法完成，由于算法需要大量的计算能力，没有access 需要排队
- A: 估计计算时间，在已有的时间内，提供最优化的解决方案；牺牲自己的时间，保证高质量完成，客户体验是最重要的，给我们funding； 我和maanger讨论，request了额外的计算权限，在短时间，将优化效果提升了很多，
- R: 最后结果不错，可能我们想要scale到1000 nodes，level，但是我们做到了500 nodes
- L: working hard， sacrifice self time, keep the high standards, make sure the product delieved on time, try our best to finish as soon as possible 

### Do you meet a task that you can not handle? why not finishing it? [Deliver Result] 
> 你是否遇到无法完成任务的时候？为什么无法完成？ - 回答问题时注意说明无法按时完成任务的原因，并且做了哪些行动上的补救，以及阐述最后的结果和影响。 

- Story: 刚来JHU选导师的事，Mike布置了一个task，一半的paper，补充完另一半，非常难，没有背景经验
- S/T: 刚来9月，11月定导师，给了一个任务，非常难，stochastic field，sampling performance KL expansion 
- A: 自学random field， stochastic process，Gaussian process，这些，写code，同时和导师沟通和请教，反复读paper
- R: 最终在10月底，做出来了，结果很好，甚至超出了导师的预期，老板给与了理解和肯定，我们这个工作，后续发表了，现在cited more than 200次


### 7. **Conflict** 分歧 - Disgree with teammate or manager [earn trust]  conflict 问题 


### 8. **Decision** 决策 - took a risk or do not have much time to make a decision [bias for action, ownership] 







### 4. Tell me about a time you had to quickly adjust your work priorities to meet changing demands 
> 调整工作优先级去满足挑战的需求



### . Tell me about a time when you were not able to meet a time commitment. What prevented you from meeting it? What was the outcome and what did you learn from it? 





Qiao Li Tips
```
Non-Technical Questions:
 
The interviewer will ask you 2-3 Non-Technical questions that are behavioral or situational questions.
The questions are based on our 14 Leadership Principles
A few sample questions may sound like:
Tell me about a time in which you disagreed with a coworker.
Tell me about your most challenging project.
Tell me about a time in which you worked under a tight deadline.
 
Tips for the Non-Technical Questions:
 
·            Understand the STAR Method (Situation, Task, Action, Result) and frame your answers in that way.
·            Go through and understand our 14 Leadership Principles. Think of a couple solid examples that would relate to them, then verbally practice this story so it’s easy to tell in the interview.
·            While telling your story, make sure you are able to explain the importance of your projects, methodologies/techniques you’ve used, and its impact to the business. They will want to know why you took the actions that you did.
·            Amazon is data driven! Make sure your stories include data points, success metrics, etc.
·            Make sure to use “I” versus “WE” as you tell your story. The interviewer wants to learn about how YOU contributed versus your team as a whole.
·            Think of at least 3-4 examples of when you excelled in your role, and maybe 1-2 examples of when you’ve failed. Failure is never easy to talk about, but interviewers want to see how you recovered from the situation or what you’ve learned from that experience to grow.
·            Try to avoid using the same example multiple times if possible. You’ll want to try to use different examples with each of the interviewers, but feel free to share an example you’ve already discussed if needed.
·            Practice behavioral questions, similar to the ones found in this article: https://www.thebalance.com/behavioral-job-interview-questions-2059620

```

来offer 亚麻BQ面试攻略: 是什么? 怎么答? 如何练习? 优秀答案长什么样? 
https://laioffer.medium.com/%E4%BA%9A%E9%BA%BBbq%E9%9D%A2%E8%AF%95%E6%94%BB%E7%95%A5-%E6%98%AF%E4%BB%80%E4%B9%88-%E6%80%8E%E4%B9%88%E7%AD%94-%E5%A6%82%E4%BD%95%E7%BB%83%E4%B9%A0-%E4%BC%98%E7%A7%80%E7%AD%94%E6%A1%88%E9%95%BF%E4%BB%80%E4%B9%88%E6%A0%B7-e5af142d4a8 



Dan's youtube 
https://www.youtube.com/watch?v=K12YTkrwolk 



1. introduce yourself

2. why amazon 

3. what's your sucessful project? 

4. difficult customer? 

5. what's your best LP？


一定要避免vague answer 没有细节的
clarity of thought 清晰的，简单的


35个问题总结
https://www.youtube.com/watch?v=ys7fLcH5gpg&t=0s






重要经验

1. 其实我觉得最重要的一条经验是是不管题目多么negative，最后的答案一定要归结到positive上 





### 高频必考 - General BQ 
- introduce yourself 

- Why Amazon? Why this team and this position? 

- Why choose industry rather than academia? 

- What's your strength and weakness? 

- What's the most interesting project? 
- What's the most challenging project? 


### 分类整理 
1. Failure：（Customer Obsession， Earn Trust ）******

2. Most challenging （Insist on the Highest Standards， Ownership ， Learn and Be Curious ） ***


3. Miss deadline （Customer Obsession，Deliver Results ，Ownership，Dive Deep）**

4. Conflict （Customer Obsession， Earn Trust ）**

5. Took a risk, or do not have much time to make a decision （Bias for Action， Ownership ）***

6. Challenges from customers. （Customer Obsession， Earn Trust ）*

7. Negative feedback （Earn Trust ）*

8. Sacrifice short for long goal （Think Big）*

9. Tough decision（Bias for Action ）*
























# Leadership principles 

### Customer Obsession 顾客至上， 全心全意为顾客服务
>Leaders start with the customer and work backwards. They work vigorously to earn and keep customer trust. Although leaders pay attention to competitors, they obsess over customers.

### Ownership 主人翁精神，不只顾着自己的一亩三分地，长期利益履行分外的职责，可以牺牲短期利益而实现长期目标
>Leaders are owners. They think long term and don’t sacrifice long-term value for short-term results. They act on behalf of the entire company, beyond just their own team. They never say “that’s not my job."

### Invent and Simplify 创新简化，多research借鉴别人的方法来提高效率
>Leaders expect and require innovation and invention from their teams and always find ways to simplify. They are externally aware, look for new ideas from everywhere, and are not limited by “not invented here." As we do new things, we accept that we may be misunderstood for long periods of time.

### Are Right, A Lot 大多数是正确的决策，努力证明自己正确，说服别人
>Leaders are right a lot. They have strong judgment and good instincts. They seek diverse perspectives and work to disconfirm their beliefs.

### Learn and Be Curious 学习工作外的事情，积极学习
>Leaders are never done learning and always seek to improve themselves. They are curious about new possibilities and act to explore them.

### Hire and Develop the Best 招人和晋升
>Leaders raise the performance bar with every hire and promotion. They recognize exceptional talent, and willingly move them throughout the organization. Leaders develop leaders and take seriously their role in coaching others. We work on behalf of our people to invent mechanisms for development like Career Choice.

### Insist on the Highest Standards 坚持高标准，不妥协自己的高要求
>Leaders have relentlessly high standards — many people may think these standards are unreasonably high. Leaders are continually raising the bar and drive their teams to deliver high quality products, services, and processes. Leaders ensure that defects do not get sent down the line and that problems are fixed so they stay fixed.

### Think Big 远见卓识
>Thinking small is a self-fulfilling prophecy. Leaders create and communicate a bold direction that inspires results. They think differently and look around corners for ways to serve customers.

### Bias for Action 崇尚行动，速度很重要，数据不足的时候，权衡利弊，想好backup plan，敢于冒险
>Speed matters in business. Many decisions and actions are reversible and do not need extensive study. We value calculated risk taking. 

### Frugality 勤俭节约
>Accomplish more with less. Constraints breed resourcefulness, self-sufficiency, and invention. There are no extra points for growing headcount, budget size, or fixed expense.

### Earn Trust 不骄傲自满，要自我批判，敢于承认错误，赢得信任
>Leaders listen attentively, speak candidly, and treat others respectfully. They are vocally self-critical, even when doing so is awkward or embarrassing. Leaders do not believe their or their team’s body odor smells of perfume. They benchmark themselves and their teams against the best.

### Dive Deep 刨根问底，有能力troubleshoot 到根本问题，解决根本问题
>Leaders operate at all levels, stay connected to the details, audit frequently, and are skeptical when metrics and anecdote differ. No task is beneath them.

### Have Backbone; Disagree and Commit 不趋附大众，敢于提出自己的意见
>Leaders are obligated to respectfully challenge decisions when they disagree, even when doing so is uncomfortable or exhausting. Leaders have conviction and are tenacious. They do not compromise for the sake of social cohesion. Once a decision is determined, they commit wholly.

### Deliver Results 达成业绩，排除万难，最后完成任务，或者最终从失败中学到了什么
>Leaders focus on the key inputs for their business and deliver them with the right quality and in a timely fashion. Despite setbacks, they rise to the occasion and never settle.

### Strive to be Earth's Best Employer 新的准则，创造更好的环境
>Leaders work every day to create a safer, more productive, higher performing, more diverse, and more just work environment. They lead with empathy, have fun at work, and make it easy for others to have fun. Leaders ask themselves: Are my fellow employees growing? Are they empowered? Are they ready for what's next? Leaders have a vision for and commitment to their employees' personal success, whether that be at Amazon or elsewhere.

### Success and Scale Bring Broad Responsibility 新的准则，世界更美好？
>We started in a garage, but we're not there anymore. We are big, we impact the world, and we are far from perfect. We must be humble and thoughtful about even the secondary effects of our actions. Our local communities, planet, and future generations need us to be better every day. We must begin each day with a determination to make better, do better, and be better for our customers, our employees, our partners, and the world at large. And we must end every day knowing we can do even more tomorrow. Leaders create more than they consume and always leave things better than how they found them.

