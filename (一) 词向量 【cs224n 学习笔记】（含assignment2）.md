# (一) 词向量 【cs224n 小白易懂学习笔记】（含assignment2）

课程地址：https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/

参考课程：Lecture1, Lecture2

预备知识：矩阵运算，链式法则求导，python

建议：

1. 把相关课程视频看完
2. 把assignment1做完
3. 尝试做assignment2, 先做在看答案
4. 点个赞

**本章大纲：**

![image-20210226165143073](C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210226165143073.png)

## 1. 如何表示文字意思

### 1.1 同义词表示

* 我们在理解或想表达一个文字/单词的意思的时候，常常会利用到与之相关的近义词或者上位词（eg: 熊猫的上位词：动物），其中```WordNet```就是一个这些词的一个词典。

  <img src="C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210223170211671.png" alt="image-20210223170211671" style="zoom:50%;" />

* 这种方法有以下问题

  <img src="C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210223170317275.png" alt="image-20210223170317275" style="zoom:50%;" />

### 1.2 独热向量表示

* 如果我们把词语看作一个离散的符号，可以用如下向量来表示一个单词。

<img src="C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210223170955678.png" alt="image-20210223170955678" style="zoom:50%;" />

​		上图我们可以理解成：

​        1. 该向量的**维度**为15，即代表这个词典里面有15个单词

​        2. motel 在第11的位置，给他个**下标**的话就是 E_11, 而hotel 则是 E_8

* 存在问题

  我们知道，motel 和 hotel 是都有相近的意思，但这两个向量是**正交**的，也就是两者没有什么关联

### 1.3 词向量表示

感觉这里讲得没有特别清晰为什么要可以用词向量来表示，下面我总结的是吴恩达的对词向量相关的理解。

* 首先，我们同样需要使用向量来表示单词，为了找到不同单词之间的相似关系，我们可以给单词每一维度附上一个特征属性，如下图：

  ![image-20210223173316684](C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210223173316684.png)

  我们提取两个向量：
  $$
  Man = \begin{bmatrix}
   -1\\
   0.01\\
   0.03\\
   0.09
  \end{bmatrix} Apple = \begin{bmatrix}
   0.00\\
   -0.01\\
   0.03\\
   0.95
  \end{bmatrix}
  $$
  

  十分好理解，Man 具有在性别方面的特征，Apple 在具有在食物方面的特征。

  **注1：** 在图中，Man下面的（5391）表示的是在有10000个单词的词典中，Man属于5391个，即在一个10000维的独热向量下, Man的独热向量是E_5391。由此，我们成功通过一个4维的向量来表示一个10000维的向量。

* 这个向量需要多少维度？每一个维度对应的特征如何确定？每一个维度对应数值如何确定？

  可能初学者会有这种类似的问题吧

  * 多少维度：

    通常应该都是有一个经验值，比如100，300这样的。

  * 每一个维度代表的是什么特征：

    上面讲的是对词向量的一种理解，证明了这种方式的向量是可以表示一个文字的，至于为什么Man的第一个维度是-1，对应的是什么，有些时候确实可以强行解释，就像**解释神经网络各个隐含层的作用**一样，但其实可能也不太需要知道。

  * 每一个维度对应数值如何确定：

    这个就是下面，也就是本章内容的**重点**啦

  **注2：**词向量又称为**词嵌入**```word embedding```, 可以这样理解，本来每个单词（假设）10000维的空间中，每个单词是一个维度。这个时候我们构造了一个300维的空间，把每个单词硬生生**嵌入**到这个空间里面去，使得这些单词之间肯定有着某些联系。



## 2. 构建/更新词向量

### 2.1 基于SVD的方式

* 首先，在语料库中采集句子，创建共现矩阵

  看例子即可：

  <img src="C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210223184516824.png" alt="image-20210223184516824" style="zoom:50%;" />

* 共现矩阵

  考虑一个新的问题： I ____ python. 

  I 后面填什么呢，我们看这个矩阵，like在I的向量当中权重最大，那么就代表最大概率应该是like。

* 相关问题：

  * 矩阵会随着词汇量而增加
  * 需要很多空间存储这个高维矩阵
  * 矩阵会有稀疏性问题

  为此，我们需要对该矩阵进行降维，降维的方法采用的是SVD奇异值分解

* SVD

  <img src="C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210223185337924.png" alt="image-20210223185337924" style="zoom:50%;" />

  如果学过机器学习当中的主成分分析（PCA）就可以很容易理解为什么降维之后的矩阵一样可以表示这个词向量。

  没学过的话，那就我们就按上面例子先定性理解一下：

  1. 先看结果，上面的矩阵是（8，8）的，我们把它降到2维，然后这个矩阵就变成了（8，2）
  2. 注意，一个单词的向量是用行来表示的（视频中也提到），具体的就是，本来like = (2,0,0,1,0,1,0,0) 变成了 （x, y）
  3. 要想一句话大概理解x,y的话，我会这样说：8个单词存在一个**8维**的空间里面，我们在里面找到了一个**最能体现这个空间维度差异的2维空间（平面）**，然后把8个单词都**映射**到这个2维平面上，映射下来之后，每个单词在**各个维度的权重**即是x和y
  4. ![image-20210223194849349](C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210223194849349.png)

* 存在的问题：
  1. 矩阵的维度会经常发⽣改变（经常增加新的单词和语料库的⼤⼩会改变）。
  2. 矩阵会⾮常的稀疏，因为很多词不会共现。
  3. 矩阵维度⼀般会⾮常⾼
  4. 基于 SVD 的⽅法的计算复杂度很⾼ ( 矩阵的计算成本是 )，并且很难合并新单词或⽂档
  5. 需要在 X 上加⼊⼀些技巧处理来解决词频的极剧的不平衡



### 2.2 Word2vec - 基于迭代学习的方式

* 两个模型算法，两个训练方法：

  * 两个模型算法：continuous bag-of-words（CBOW）和 skip-gram
  * 两个训练⽅法：negative sampling 和 hierarchical softmax

  本章主要介绍skip-gram 和 negative sampling

#### 2.2.1 skip-gram 语言模型

* Word2Vec 中有一个核心的概念：

  > “You shall know a word by the company it keeps”

  ![image-20210223200112579](C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210223200112579.png)

  即一个单词的意思由整个句子来决定

* 有一个句子“The cat jumped over the puddle.” 这个是一个完整有效的句子，我们要构建一个语言概率模型P(W1,W2,...Wn)，假设P(Wt)为第t个单词出现的概率，我们要令该概率变得很大，其中：
  $$
  P(w_1,w_2,...,w_n) = \prod_{t=1}^{n} P(w_t)
  $$
  但有一点比较荒唐，P(Wi)怎么确定的，完全没有厘头，这个时候，我们就要用上这个单词与上下文的联系，这个时候就可以用到下图：

  ![image-20210223201317425](C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210223201317425.png)

  这个时候，我们的P(Wt)变成了：
  $$
  P(W_t) = P(W_{t-2}|W_t)*P(W_{t-1}|W_t)*P(W_{t+1}|W_t)*P(W_{t+2}|W_t)
  $$
  

  也就产生了下面的公式：

  <img src="C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210223201708324.png" alt="image-20210223201708324" style="zoom:50%;" />

  就是让这个句子是个正确的句子的概率

* 我们看这个P, 就是已知中心词的情况下，上下文词的概率，skip-gram 就是根据中⼼词预测周围上下⽂的词。相反，CBOW 是根据中心词周围的上下文单词来预测该词。

#### 2.2.2 损失函数

* 损失函数

  我们的目标就是让上面的这个概率变大，为了计算，我们转化成以下公式：

  <img src="C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210223202217760.png" alt="image-20210223202217760" style="zoom:50%;" />

  负号是为了转化成最小值问题，log 则是为了连乘变成求和

* P

  <img src="C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210223202527254.png" alt="image-20210223202527254" style="zoom:50%;" />

  * 这个是softmax的概率模型

    ![image-20210223202738351](C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210223202738351.png)

  * v_c 是中心词向量，按上例则是”into“的词向量
  * u_o是上下文词向量，按上例，选其中一个，比如”turning“的词向量
  * 两者进行点乘，如果两者相似度比较高的好，那么这个值就会比较大
  * 分母则是为了使结果保证在0~1，形成一个概率分布

#### 2.2.3 梯度下降

* 接下来我们就用梯度下降的方法来让我们的损失函数达到最小值

  <img src="C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210226111713154.png" alt="image-20210226111713154" style="zoom:50%;" />

* 这些都默认大家已经知道啦，不知道可以看看吴恩达的机器学习

  <img src="C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210226111741365.png" alt="image-20210226111741365" style="zoom:50%;" />

* 我们的目标就是不断**update** u_o 和 v_c, 通过更新这两个词向量使得损失函数不断减少，为此，更新后的词向量可以更好的表达这个单词的意思。

* 下面是两个偏导数的求导

  <img src="C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210226112046479.png" alt="image-20210226112046479" style="zoom:50%;" />

  <img src="C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210226112101326.png" alt="image-20210226112101326" style="zoom:67%;" />

---

<img src="C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210226112120246.png" alt="image-20210226112120246" style="zoom:50%;" />

* 而为了加快下降速度，我们会采用随机梯度下降的方式（SGD）

  <img src="C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210226112227544.png" alt="image-20210226112227544" style="zoom:50%;" />

#### 2.2.4 negative sample

* 2.2.2 中采用的损失函数是基于softmax的，但其中用于归一化的分母计算量太高，需要计算所有的上下文词向量也就是**U**, 所以，在Word2vec当中会改进这个方法，使用的是负采样的方法。

  <img src="C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210226113049627.png" alt="image-20210226113049627" style="zoom:50%;" />

* 什么叫**负采样**呢？

  * 拿个上面的例子

    ![image-20210226112935436](C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210226112935436.png)

    假设window_size = 4, 那么所谓**正采样**就是把“turning” "into" "crises" "as" ，这四个单词的词向量与中心词“banking”的词向量相乘，使得这两者向量点积更大，所以就有上面公式的前面一项。

    那我们还要更新“banking”这个单词，使得它跟令外一些单词应该是没有关系的，比如“government”，也就是让这个“banking” 和“government” 两个词向量关系更小一点，所以就有了 -u_k^Tv_c . 

    那我们要拿几个没关系的词向量呢，拿 **k** 个

    那我们该**怎么选择**这些无关的单词呢，采用如下公式：

    <img src="C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210226113609302.png" alt="image-20210226113609302" style="zoom:50%;" />

    P(w) 越大，我们就选它

---

### 2.3 Glove - 结合上述两者的模型算法

由于作业没有遇到，暂时我也只是定性地理解，如果之后有用到会补充，其他资料见[CS224N笔记(二)：GloVe](https://zhuanlan.zhihu.com/p/60208480)

---

## 3. 评估词向量

### 3.1 词类比

* 我们如果评估**一个词向量是否真的能很好的表示这个单词呢**？我们这个时候看回1.3 词向量表示的例子。

<img src="C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210226114207950.png" alt="image-20210226114207950" style="zoom:50%;" />

​		我们将“Man”的embedding vector与“Woman”的embedding vector相减：

<img src="C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210226114348975.png" alt="image-20210226114348975"  />

​		类似地，我们将“King”的embedding vector与“Queen”的embedding vector相减：

​											![image-20210226114427047](C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210226114427047.png)

​		相减结果表明，“Man”与“Woman”的主要区别是性别，“King”与“Queen”也是一样。

​		那么就能说明这四个词向量是很好的

* 我们再能看下面这个问题

  ![image-20210226114639212](C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210226114639212.png)

  我们可以知道，这个问号应该就是queen, 那么我们怎样才能找到这个queen的词向量呢。

  我们采用以下公式```cosine 相似函数```找到xi：

  ![image-20210226114754806](C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210226114754806.png)

  也就是说在我们的词汇库找X, 然后找到最大的xi ,使得d最大，如果这个xi 刚好就是queen的词向量，那么就表示我们训练出来的词向量是很好的啦 ：）



---

## 4. Assignment2

### 4.1 手写部分

下面是书写部分的前三题，主要为了对y, y^, U, V 有更好的理解，之后的题目答案详见[CS224n-2019 Assignment](https://looperxx.github.io/CS224n-2019-Assignment/#assignment-02)

<img src="C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210226120438083.png" alt="image-20210226120438083" style="zoom:50%;" />![image-20210226120445758](C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210226120445758.png)



<img src="C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210226120455591.png" alt="image-20210226120455591" style="zoom:50%;" />

​	

<img src="C:\Users\95152\AppData\Roaming\Typora\typora-user-images\image-20210226120504799.png" alt="image-20210226120504799" style="zoom:50%;" />



### 4.2 代码部分

* word2vec.py

  * sigmoid 函数

    ```py
    s = 1 / (1 + np.exp(-x))
    ```

  * softmax 损失函数

    ```python
    scores = np.dot(outsideVectors, centerWordVec)   # (vocab_size,1)
    probs = softmax(scores)                          # (vocab_size,1)  y_hat
    
    loss = -np.log(probs[outsideWordIdx])
    
    dscores = probs.copy()   # (vocab_size,1)
    dscores[outsideWordIdx] = dscores[outsideWordIdx] - 1   #  y_hat minus y
    gradCenterVec = np.matmul(outsideVectors.T, dscores)  # (embedding_dim,1)
    gradOutsideVecs = np.outer(dscores, centerWordVec) # (vocab_size,embedding_dim)
    ```

  * negative 损失函数

    ```python
    gradCenterVec   = np.zeros(centerWordVec.shape)
    gradOutsideVecs = np.zeros(outsideVectors.shape)
    loss = 0.0
    
    u_o = outsideVectors[outsideWordIdx]
    z = sigmoid(np.dot(u_o, centerWordVec))   # (vocab_size,1)
    loss = - np.log(z)
    gradCenterVec = (z - 1) * u_o
    gradOutsideVecs[outsideWordIdx] = (z - 1) * centerWordVec
    for i in range(K):
        neg_id = indices[i+1]
        u_k = outsideVectors[neg_id]
        z = sigmoid(-np.dot(u_k,centerWordVec))
        loss -= np.log(z)
        gradCenterVec += u_k*(1-z)
        gradOutsideVecs[neg_id] += centerWordVec*(1-z)
    ```

  * skipgram函数

    ```python
    center_id = word2Ind[currentCenterWord]
    centerWordVec = centerWordVectors[center_id]
    for word in outsideWords:
        outside_id = word2Ind[word]
        loss_mini, gradCenter_mini, gradOutside_mini= \
        word2vecLossAndGradient(centerWordVec=centerWordVec,
                                outsideWordIdx=outside_id,outsideVectors=outsideVectors,dataset=dataset)
        loss += loss_mini
        gradCenterVecs[center_id] += gradCenter_mini
        gradOutsideVectors += gradOutside_mini
    ```

    

  * sgd 函数

    ```python
    loss, gradient = f(x)
    x = x - step * gradient
    ```

---

## 5. 参考资料

1. [CS224n-2019 笔记列表](https://zhuanlan.zhihu.com/p/68502016)



