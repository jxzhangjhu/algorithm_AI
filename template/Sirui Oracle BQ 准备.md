BQ 鸡精

0. Introduce yourself, strength and weakness? 
1. why Oracle?
2. Why are you leaving your current role?
3. What do you hope to gain in your new role? What's your ideal role?
4. How would you solve the conflict with your co-worker? 
5. What's your most challenge project?  What's your pride of your project? 
6. Do you have any exeprience in last minute change/roll back? 
7. Meet customer requirement exeprience
8. How do you handle the co-worker you don't like
9. Mistake and how to address? 
10. Tight deadline
11. Do you have experience to learn a new tech never studied? 
12. before other people find the problem, you solve the problem first? 
13. how do you hanlde the case that you are assigned by multiple projects? how do you solve if there are some unfair treatment? 
14. What's most like core value and dislike one? 


# 22 Questions

## 1. Negative feedback 不好的反馈 *Earn Trust*

### 22 Questions Asked in 94% Amazon Interviews
7. Tell me about a time when you received negative feedback from your manager. How did you respond? - done!

>你曾收到过来自同事或领导的负面反馈吗？你是如何改进的？ - 个人都有犯错的时候，收到负面反馈并不是难以启齿的事情, 把为什么收到负面反馈的原因说清，并且总结从中学到的经验、重新得到他人的认可才是重点。

- Stroy: Image-to-image regression project, 开始的时候没有很好地理解domain problem，导致project review feedback不是很好
- S/T : 来ORNL开始的时候，项目不太了解，image-to-image regression, in AM  时间比较紧，我开始独立做，但是缺少background，review结果不是很理解，其实我应该早点寻求LDRD project manager的帮助， 说我应该早点寻求帮助
- A: 我意识到这个问题， 合理的评估自己的困难，找寻合适的domain expert，他来帮助我生成data，并且理解data，并且和project manager交流进度, domain knowledge is important 
- R: 我学会了自我评估和及时沟通，最终这个project 效果很多，我们发表了2篇论文，program manager很满意，给与了肯定. 


## 2. Collabration, teamwork, interpersonal skills 合作 

### Your team/co-workder/collabroator does not perform well, have bad impact to your work, how to handle that? [Ownership]
>如果你的小组同事经常表现不好，甚至影响到你工作，您是如何处理的？ - 因为团队是需要合作的。找到队友犯错的原因，帮助解决问题，完成工作任务才是重点。

> MDF story Sirui也可以用！ 
- Story: LDRD project, Anomly detection with MDF, 明确他们的困难，还有credit 分配的问题，实现共赢！
- S/T: 他们不来meeting，比较缓慢，积极性不高
- A: 首先沟通了解困难和问题，发现他们是时间还有credit，他们想从应用方面体现，时间上和manager沟通调整我们的timeline，然后credit，我们从AI和计算方面证明，他们从manufacturing
- R: 效果很不错 - 我们的paper发表在neurips，已经一个dataset， 他们成功上线我们的方法，节约了大量人工标注的成本，提高了工作效率，5-10% performance 提升
- L: 学习到了沟通的重要性，同时协同能力，如何实现win-win的principle

### 22 Questions Asked in 94% Amazon Interviews
2. What did you do when you needed to motivate a group of individuals?
> AD project，和不同domain的人合作，很重要是了解他们的expectation，比如从domain角度证明credit，来设定一个win-win的goal，这样我们更有motivation

3. Give me an example of a time you faced a conflict while working on a team. How did you handle that?
> 仍然是conflict 的问题，case 3 - AD+ AM problems，去做semi-supervised AD, 他们非要定义成supervised classification，他们的label有错误，事实证明他们的机器label确实有问题

4. The last time you had to apologize to someone
> 和phd 老板开会的事，最后时间发现boundary condition弄错了，要改，但是结果是好的！ 

5. Describe a long-term project that you managed. How did you keep everything moving along in a timely manner?
> AI for AM这个project，从19年实习开始一直在做； 从regression, anomaly detection and reconstruction，这个故事比较系统，了解了CV的不同的方向

6. Describe a situation when you negotiated with others in your organization to reach an agreement.
> imbalance regression的那个problem，和guannan zhang，我认为需要under sampling，他们认为不需要，作为regression， MDF那个项目





## 3. Failure/mistake 最大的失误和失败 [如果onsite多准备一些！准备five strong ones, 至少1-2个 for phone]

### Tell me about a time you failed/the biggest mistake you made 
> 最大的失误和失败

- Story: Ph.D. 开会的事，当时Boundary condition 弄错了，重新弄最终赶上了并且在present之前搞定了
- S/T: Ph.D. 去开会，即将做presentation，但是发现BC错了，要改
- A: 我重新check所有的BC然后request simulation licence，run whole night改过来了
- R: 顺利赶上了我的talk，得到很好地反馈，尽管过程很struggle
- L: 需要提前check，和导师多沟通



### 22 Questions Asked in 94% Amazon Interviews

18. Tell me about a time when you were not able to meet a time commitment. What prevented you from meeting it? What was the outcome and what did you learn from it?

> 本质就是missing deadline, 讲一下实习的时候AI expo的stroy，要给talk，来不及做了！

- S/T：我们当时想到了一个很好的idea，graph learning and generative models for materials design, 当时还有3个月的时间，去年10.5，我觉得我们可以赶得上。 我的任务是利用graph来生成新的materials，我没能及时搞定一个baseline，而是一直对最复杂的case，来追求最好的效果，但事实证明framework有一些问题，问题太复杂，不知道那个地方出现了问题，所以最终没有赶上
- A: 在这之后，我改变了方针，首先无论距离下一个deadline还有多久，我们要提前规划好timeline， step-by-step的move forward because time goes very fast; second，对于复杂的问题，我需要divide and conquer，先解决简单的问题，我们当时要同时生成node, connection and lattice paramters, 我们分别设计了3个case，来分别生成这些，当code works well，我们再去解决最复杂的case; third, a good lession 这只是一个conference 后面还有其他的，但是将来对于costomer，可能还有下一个costomer也可能没有，所以一定要避免再次发生. 

19. Tell me about a time you failed and what you learned from it
> 不能把自己说的太差了，关键是别人的问题导致了你的failure。 why then want to ask this questions? 是讲故事的能力，非常重要， 你是否容易相处，是否从中学习到了

- story: 申请faculty，拿到几个onsite，最终没有拿到, 方向很重要，选择比努力更重要， 很远原因决定了
- S/T: apply many positions and received several phone interview and onsite inviation but did not get an offer
- A: I found they expect more domain specific, but interests more on AI + AM 
- R: I did not get but I found my interests may not be in academic so I transfer to industry, choose is more important than working hard 






## 4. Challenge 挑战 [Customer obsession] 最重要的

### Most challenging project [highest standards, think big]
> 如何走出comfort zone，如何提升自己

- Story: 可以是AD，image-to-image regression, deep learning accelerated AI, phd做的东西？ 
- S/T: 主要解释为什么challenging，主要是因为之前没做过，这个问题不容易，同时要求比较高， 时间比较短
- A: 如何解决的呢 1）学习ML相关的知识，看paper，review paper；(2) implementation，coding， hpc， end-to-end design （3）communication，domain expert，collaborated with MDF
- R: deeper understanding with publications 


### 22 Questions Asked in 94% Amazon Interviews
17. We all deal with difficult customers from time to time. Tell me about a challenging client-facing situation and how you handled it.
> 不好搞的顾客怎么办？

- Story: 还是MDF的合作; 关键点是 why difficult, why challenging? where is the challenging? 
- S/T: difficult 他们能提供的data很有限，因为成本高，不愿意付出更多时间来label data，提供的信息有限； 挑战就是在有限data下如何做？
- A: set-up meeting, 寻求manager帮助，多沟通得到更多支持和信息
- R: 合作很愉快，产品上线，发表paper 




## 5. Deadline 

### Missing deadline [customer obsession, deliver results]
> 要突出如何让影响最小，让customer不受影响，为何会miss，最后结果是好的, 影响不是很大

或者用phd开会的例子？ 

- Story: 用DGS做一个demo，topology optimization for MDF，manager 突然说customer 提前1周时间去看这个demo，没办法搞定，需要超算
- S/T: deadline 提前，无法完成，由于算法需要大量的计算能力，没有access 需要排队
- A: 估计计算时间，在已有的时间内，提供最优化的解决方案；牺牲自己的时间，保证高质量完成，客户体验是最重要的，给我们funding； 我和maanger讨论，request了额外的计算权限，在短时间，将优化效果提升了很多，
- R: 最后结果不错，可能我们想要scale到1000 nodes，level，但是我们做到了500 nodes
- L: working hard， sacrifice self time, keep the high standards, make sure the product delieved on time, try our best to finish as soon as possible 

### Do you meet a task that you can not handle? why not finishing it? [Deliver Result] 
> 你是否遇到无法完成任务的时候？为什么无法完成？ - 回答问题时注意说明无法按时完成任务的原因，并且做了哪些行动上的补救，以及阐述最后的结果和影响。 

- Story: yes 重点是无法完成的原因， 这个可以说说phd做实验，赶paper review 补实验进度，做demonstrate，要在一天之内做很多样本， 因为3d打印很慢
- S/T：就是描述这个问题的关键， 为什么无法完成，原因是什么？ 一定要把这个讲清楚！ 
- A：采取的措施是不是make sense？ 借其他组的打印机一起打，同时寻求组里或者其他collaborator一起帮忙
- R: 顺利赶上deadline，paper最终接受了
- L: 需要提前规划时间，特别是有deadline的时候








## 6. Conflict 分歧 

### Disgree with teammate or manager [earn trust]  conflict 问题 
or when you did something without asking approval from your manager? 
> 不是强行谁付别人或者什么没有理由顺应比人，而是为了customer或者更重要的目的所以不同意别人。介绍背景，说出分歧，然后说自己怎么做，可以是讨论trade off

- Stroy: 分歧是什么？ simulate 这个lattice，用solid 还是 beam element，和我的teammate
- S/T: 和advisor多讨论，寻求建议，
- A：多做模拟，同时做实验来验证，我们的结果和真实的比较， 我们可以有一些tradeoff，比如连接点用solid，特殊处理，兼顾2者的优势
- R: 结果很好，导师很满意，我们的讨论，在我们的paper我们也给出了2种方法，并且比较各自的优缺点！ 






## 7. **Action, Decision and Responsibilities** 决策和责任 [*bias for action*]

### Took a risk or do not have much time to make a decision [bias for action, ownership] 
or tell me about a time when you had to work on a project with unclear responsibilities？
or tell me a time when you took on something significant outside your area of responsibility 
> 行动优先，当仁不让，有责任能顶上的意识，比如customer找不到人刚好你在，你能主动做些事

- Story: 很多人想要用microCT machine，但需要有人培训，但负责培训的人不在，我很了解这个工作，虽然超出我的范围，我仍然帮助他们train了
- S/T: summer的时候，虽然不是我的工作，我很忙，但还是帮助他们training
- A: 我和他们积极的合作，帮助他们设计样本，了解machine
- R: 我学到了很多，虽然比较忙，他们有各种各样的样本，我更感兴趣CV了

### 22 Questions Asked in 94% Amazon Interviews
1. Tell me about a time you had to quickly adjust your work priorities to meet changing demands 
> 主要是考察抗压能力！
- story: 和上面的类似我觉得！ 调整当时的工作来适应training 其他人的任务



10. Tell me about a time when you were 75% through a project, & you had to pivot strategy to ensure successful delivery
> 类似上面的问题？ 

11. Tell me about a time you had to deal with ambiguity
> 可以说说image recosntruction, in healthcare 这个 - 重点在于什么是ambiguity？ data少，noisy， 我把他clear这个问题，就是uncertainty quantification + reconstruction！
- story: 现在lab的一个需求是做healthcare image reconstruction，他们只有非常有限的data，希望得到准确的可靠的image
- A: literature review， 收集所有相关的paper，看是否有相关的办法；考虑他们的局限，small data， reliability，需要考虑不确定性，在这个基础上，我们提出了robust flow-based method 去解决这个问题
- R: 效果挺不错的，发了paper，同时还有其他可以用的应用，包括biomedical and healthcare 

12. Tell me about the toughest decision you've had to make in the past six months
> 从lab or academic 跳到industry吧，走出自己的confort zone，在lab 总体还是比较stable，但是考虑long-term career，我认为industry更好的选择

13. Tell me about the data-driven decision strategy [不怎么考]
> 读paper，选方向， model

 



## 8. Simplify 简化和创新 / learn and be curious 

### Tell me about a time when you gave a simple solution to a complex problem/find a new way to do something 
> 问题一定要让面试官信服你的例子，先说一般的方法很耗时，很难，自己找到一个方法很快就完成并且delever了，而且很稳定，没有过问题，结果要是好的

- Story: transfer learning 我觉得可以
- S/T: 问题比较复杂，data有限，transfer learning is a good way 这个idea 不错！ 


### 22 Questions Asked in 94% Amazon Interviews
13. What’s the most innovative new idea that you have implemented? [就用上面的就行]









## 9. Dive Deep / Problem solving 

### 22 Questions Asked in 94% Amazon Interviews
8. Tell me about a time when you missed an obvious solution to a problem 
> data augumentation and transfer learning ? image-to-image regression? 可以作为潜在的答案


9. A time when you faced a problem that had multiple possible solutions
> story - anomaly detection, supervised, unsupervised, and semi-supervised, 我为什么选择了最后？



## 10. Ownership 第二重要的！

### 22 Questions Asked in 94% Amazon Interviews
14. Tell me about a time you stepped up into a leadership role
- AI for AM 这个大方向： （1）intern 做design，用AI加速AM 设计; (2) AI 加速simulation; (3) AI + anomaly detection for production quality improvement 


15. Describe a time when you sacrificed short term goals for long term success
- 可以讲一下，短期intern结束之后，可以发表paper，但可能需要一段时间，但我和我mentor坚持认为要继续开发这个算法，然后申请funding，长期成功是proposal中了，2年的DOE ASCRfunding，support我的postdoc，虽然当时错过了一些paper deadline，但后续我们更有solid的结果和work

16. Tell me about a time when you had to push back to HQ or challenged a decision
- 不知道啥意思











--- 

Sirui Oracle BQ 准备


## 1. Put customers first 顾客至上 (Customer Obsession - Amazon)

We exist to satisfy our customers. We do this by listening to them carefully, responding to them promptly, advising them honestly, and exceeding their expectations. We put doing the right thing for customers ahead of doing what they specifically say or ask for. When faced with a choice between what is easy for us and what is good for customers, customers win every time.


## 2. Act now, iterate 行动和执行力 (Bias for Action, deliever results - Amazon)

We favor action. Notice something that needs fixing? Fix it. See a gap? Fill it. Struggle with a bad process? Improve it. See room for improvement? Grow. We move quickly but deliberately, and we iterate toward better solutions. We recognize that a grungy solution now is superior to no solution at all. We keep it simple. We don’t discuss endlessly, and we are scientific in our approach. We offer solutions, not problem statements.

## 3. Nail the basics 扎实基础 (Dive deep - Amazon)

We focus on fundamentals over flash. We recognize when we don’t have the basics in place, and we diligently work to fill the gaps. We recognize that the path to advanced solutions always runs through the basics. We focus our conversations and our products on what is currently appropriate. We make forward progress despite not having complete information or perfect solutions.

## 4. Expect and embrace change  拥抱改变，不拘泥现在 (Invent and Simplify - Amazon)

We accept change as a given. We value people who align quickly with current priorities, who have situational awareness, and who are willing to adapt. We are not limited to priorities we set in the past. We do not hang on to outdated processes and goals, and we promote or accept new ideas fearlessly. We embrace change as an opportunity for growth and greater success.

## 5. Innovate Together 创新和多样性，一起成功 （这个对应amazon 什么呢？）

We practice empathy and respect in our interactions.
We challenge our internal biases and learn to be more effective allies. We seek and celebrate the diverse perspectives and thought leadership that add value to our workplace, teams, and products. We foster an inclusive environment where collaboration drives innovation. We provide the tools, processes, and resources to enable everyone to reach their full potential and find success together.

## 6. Take risks, remain calm 承担风险，保持冷静, 合理应对  （这个对应amazon 什么呢？）

We take risks because they are necessary to our success; not taking risks is the biggest risk of all. We are logical and data-driven in assessing our risks. We react to unexpected situations by remaining calm, and then making and executing mitigation plans. We recognize that learning from our failures is part of our path to success.

## 7. Own without ego 不要以自我为中心 (Ownership - Amazon)

We take responsibility for the state of our team, our products, and ourselves. We champion the ideas we believe in. We welcome all help and feedback, and we recognize and incorporate the best ideas offered. We are the first to admit when we are wrong. We believe that our team can produce far more together than we can as lone individuals. When we notice a problem we either fix it ourselves or find another owner. We never say, “That’s not my job.”

## 8. Earn trust, give trust 获得信任，自我批评，敢于承认错误 (Earn Trust - Amazon)

We build trust by communicating openly and transparently. We give trust easily, and we recognize that trusting each other is essential to our success. We act responsibly, and we trust others to also be responsible. We don’t let occasional failures and differences in work styles undermine our trust. We learn from failures rather than seeking to place blame, and we don’t invoke rank to convince others we are right.


## 9. Take pride in your work 为你自己的工作而自豪 （这个对应amazon 什么呢？） 

We strive for excellence in all that we do, and we take pride in our progress. We do our best when we are proud of what we do. We identify work that needs to be done to achieve our team goals, and we communicate those goals well to the broader organization. We take responsibility
for either changing our work or changing ourselves when we don’t find pride in our work. We invest broadly in things that allow us to excel at our jobs, dominate in the marketplace, and delight our customers. We achieve things of value, and we value our achievements.


## 10. Challenge ideas, champion execution 喜欢质疑，批判性思维，执行力 （highest standards, have backbone, 不妥协自己的高要求，不趋附大众，敢于提出自己的意见）

We ask why. We test the validity of ideas through rigor, research, and critical thinking. We have an obligation to create clarity for ourselves and others. We own decisions. We drive effective implementation.




