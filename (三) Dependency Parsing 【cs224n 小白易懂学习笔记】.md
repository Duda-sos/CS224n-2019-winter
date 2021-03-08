# (三)  依存句法分析，pytorch【cs224n 小白易懂学习笔记】（含assignment3）

课程地址：https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/

参考课程：Lecture5

预备知识：神经网络，python

建议：

1. 把相关课程视频看完
2. 把assignment3做完（本人做的是2020版的）
3. 点个赞

本节大纲：



---



## 1. 为什么要句法分析

* 我们需要理解句子的结构，这样才能更好更正确地解释语言
* NLP中解析句子的结构有两种：短语结构和依存结构

## 2. 什么是依存句法分析

### 2.1 依存结构

* 句⼦的依存结构展示了单词依赖于另外⼀个单词（修饰或者是参数）。词与词之间的⼆元⾮对称关系称为依存关系，描述为从 head（被修饰的主题）⽤箭头指向 dependent （修饰语）。⼀般这些依存关系形成树结构。他们通常⽤语法关系的名称（主体，介词宾语，同位语等）。⼀个依存树的例⼦如下图所示：

  ![image-20210308001210057](../github/image/image-20210308001210057.png)

### 2.2 基于转移的句法分析

* 详细的定义如下：（建议照着例子来看）

  ![image-20210308001507587](../github/image/image-20210308001507587.png)

* 例子1：教程中的 “I ate fish”

* 例子2：[5.3  Neural Transition-Based Dependency Parsing](#jump)

## 3. 基于神经网络的依存句法分析

* 把3种输入特征转换为词向量
  * stack和buffer中的单词及其dependent word
  * 单词的part-of-speech tag
  * 描述语法关系的arc label
* 将它们联结起来作为输⼊层，再经过若⼲⾮线性的隐藏层，最后加⼊softmaxlayer得到shift-reduce解析器的动作（即shift\LA\RA）

![image-20210308155132238](../github/image/image-20210308155132238.png)

* 模型结构

  ![image-20210308160233693](../github/image/image-20210308160233693.png)

* 评价标准
  * Unlabeled attachment score (UAS) = head
  * Labeled attachment score (LAS) = head and label

## 4. 基于pytorch的词窗口分类

下面是我对2021年中对pytorch的tutorial进行的翻译，读完之后对assignment3的完成会有很大帮助



## 5. assignment 3

### 5.1 Adam Optimizer

* ![image-20210302154534955](D:\Learning\nlp\cs224n\github\image\image-20210302154534955.png)

  * 这种计算方法称为**指数加权平均**

    相关笔记可见[吴恩达《优化深度神经网络》精炼笔记（2）-- 优化算法](https://mp.weixin.qq.com/s/PelKW51AbMBcqOTff_JliA)

  * eg. 当 b1 设为 0.9 的时候，相当于当前的变化量是前10个不同时间变化量的指数加权平均
  * q1: 也就是说每一点的梯度都与前面时间的梯度有关系，那么当前这一点的梯度必然也和前面梯度的方向有一定的一致性，从而“stop the updates from varying as much”
  * q2: 通过指数加权平均，使得梯度下降更加稳定，平滑，减少minibatch 的振荡，从而可以更快地到达最低处，即“be helpful to learning”

* ![image-20210302155346437](D:\Learning\nlp\cs224n\github\image\image-20210302155346437.png)

  * q1: 在神经网络中，需要被更新的参数其实就是w 和 b, 所以其中一个参数的变化量相对小的化，那么他就会得到“larger updates”
  * q2: 通过这个，一方面，可以使得变化量大，即有较大震荡的参数，更新得更加平滑；另一方面，变化量小，梯度较缓和的，可以增大梯度。从而达到快速且平稳的下降。

### 5.2 Dropout

* ![image-20210302160915804](D:\Learning\nlp\cs224n\github\image\image-20210302160915804.png)

  

  ![image-20210302160924712](D:\Learning\nlp\cs224n\github\image\image-20210302160924712.png)

  * ![image-20210302163855734](D:\Learning\nlp\cs224n\github\image\image-20210302163855734.png)
  * 如果我们在评估期间应用 dropout ，那么评估结果将会具有随机性，并不能体现模型的真实性能，违背了正则化的初衷。通过在评估期间禁用 dropout，从而观察模型的性能与正则化的效果，保证模型的参数得到正确的更新。

---

### 5.3 Neural Transition-Based Dependency Parsing<a name="jump"></a>

![image-20210302192442611](../github/image/image-20210302192442611.png)

| Stack                          | Buffer                                 | New dependency   | Transition            |
| ------------------------------ | -------------------------------------- | ---------------- | --------------------- |
| [ROOT]                         | [I, parsed, this, sentence, correctly] |                  | Initial Configuration |
| [ROOT, I]                      | [parsed, this, sentence, correctly]    |                  | SHIFT                 |
| [ROOT, I, parsed]              | [this, sentence, correctly]            |                  | SHIFT                 |
| [ROOT, parsed]                 | [this, sentence, correctly]            | parsed→I         | LEFT-ARC              |
| [ROOT, parsed, this]           | [sentence, correctly]                  |                  | SHIFT                 |
| [ROOT, parsed, this, sentence] | [correctly]                            |                  | SHIFT                 |
| [ROOT, parsed, sentence]       | [correctly]                            | sentence→this    | LEFT-ARC              |
| [ROOT, parsed]                 | [correctly]                            | parsed→sentence  | RIGHT-ARC             |
| [ROOT, parsed, correctly]      | []                                     |                  | SHIFT                 |
| [ROOT, parsed]                 | []                                     | parsed→correctly | RIGHT-ARC             |
| [ROOT]                         | []                                     | ROOT→parsed      | RIGHT-ARC             |

![image-20210302193616504](../github/image/image-20210302193616504.png)

​	包含nn个单词的句子需要 2×n步才能完成解析。因为需要进行 n 步的 SHIFTSHIFT 操作和 共计n 步的 LEFT-ARC 或 RIGHT-ARC 操作，才能完成解析。（每个单词都需要一次SHIFT和ARC的操作，初始化步骤不计算在内）

### 5.4 代码部分



## 6. 参考资料

1. [CS224n-2019 笔记列表](https://zhuanlan.zhihu.com/p/68502016)
2. [CS224n-winter-together](https://github.com/Duda-sos/CS224n-winter-together)

### 