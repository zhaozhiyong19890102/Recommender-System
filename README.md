> 欢迎大家关注个人网站**www.felixzhao.cn**

# Recommender-System论文、学习资料以及业界分享

推荐系统（Recommender System）是机器学习应用较为成熟的方向之一，在工业界中，推荐系统也是大数据领域成功的应用之一。在一个较为完整的推荐系统中，不仅包含大家熟知的召回和排序两个阶段的常用算法之外，还会涉及到内容理解的部分的相关算法。除了算法之外，还涉及到大数据相关的处理技术以及工程实践，以下总结和整理以工业界的推荐系统为例，包含如下的几个部分：
- 推荐系统的概述及其技术架构
- 召回排序算法
- 内容理解
- 架构工程实践
- 工业界解决方案

**(以下内容会持续更新)**

# 1. 推荐系统的概述及其技术架构

- [2005]. [Toward the Next Generation of Recommender Systems A Survey of the State-of-the-Art and Possible Extensions](https://ieeexplore.ieee.org/document/1423975)
  - 简介：据说是推荐系统的必读文章，2005年的state-of-the-art的推荐综述，按照content-based, CF, Hybrid的分类方法进行组织，并介绍了推荐引擎设计时需要关注的特性指标，内容非常全。

- [2010]. [The YouTube video recommendation system](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Reference/The%20YouTube%20video%20recommendation%20system.pdf)
  - 简介：2010年的YouTube的推荐系统的文章，文章中还没有涉及到高大上的算法，但是在文章中，值得我们借鉴的是对推荐系统的理解，包括产品的理念，数据的处理，系统的设计，是一篇值得学习的实践性的文章，建议认真研读。

- [Netflix公布个性化和推荐系统架构](http://www.infoq.com/cn/news/2013/04/netflix-ml-architecture "Netflix公布个性化和推荐系统架构")
  - 简介：介绍在Netflix中的推荐系统的技术架构，对于构建工业级的推荐系统具有很重要的意义。

- [Overview of Recommender Algorithms – Part 1](https://buildingrecommenders.wordpress.com/2015/11/16/overview-of-recommender-algorithms-part-1/ "Overview of Recommender Algorithms – Part 1")

- [Overview of Recommender Algorithms – Part 2](https://buildingrecommenders.wordpress.com/2015/11/18/overview-of-recommender-algorithms-part-2/ "Overview of Recommender Algorithms – Part 2")

- [Recommender Systems in Netflix](https://buildingrecommenders.wordpress.com/2015/11/18/recommender-systems-in-netflix/ "Recommender Systems in Netflix")

- [推荐引擎初探](https://www.ibm.com/developerworks/cn/web/1103_zhaoct_recommstudy1/index.html#icomments "推荐引擎初探")

- [深度学习在推荐领域的应用](http://geek.csdn.net/news/detail/200138 "深度学习在推荐领域的应用")

- [2017]. [Deep Learning based Recommender System : A Survey and New Perspectives](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Reference/Review/Deep%20Learning%20based%20Recommender%20System%20A%20Survey%20and%20New%20Perspectives.pdf)

- [2017]. [Use of Deep Learning in Modern Recommendation System : A Summary of Recent Works](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Reference/Review/Use%20of%20Deep%20Learning%20in%20Modern%20Recommendation%20System%20A%20Summary%20of%20Recent%20Works.pdf)

- [2020]. [从零开始了解推荐系统全貌](https://zhuanlan.zhihu.com/p/259985388)
  - 简介：介绍了推荐系统的多个方面，包括用户画像，召回排序算法及策略，比较全面，适合初学者了解推荐系统的全貌
  - 推荐指数：\* \* \*

# 2. 召回排序算法

- [2000]. [Application of Dimensionality Reduction in Recommender System -- A Case Study]()
- [2003]. [Amazon.com recommendations:Item-to-item collaborative filtering]()
  - 简介：经典的亚马逊item-based算法的文章
- [2007]. [Predicting clicks: estimating the click-through rate for new ads]()
  - 简介：LR算法应用于CTR问题
  - 推荐指数：\* \* \* \* \*
- [2010]. [Factorization Machines]()
  - 简介：FM算法在CTR中的应用
  - 推荐指数：\* \* \* \* \*
- [2010]. [Fast Context-aware Recommendations with Factorization Machines]()
- [2013]. [Learning deep structured semantic models for web search using clickthrough data]()
  - 简介：本文提出了DSSM模型，在原始论文中，最初是用于在搜索中计算用于是否点击的，现通常被应用在推荐中的相似性召回
  - 推荐指数：\* \* \* \* \*
  - 阅读笔记：[深度语义模型DSSM](http://felixzhao.cn/Articles/article/4)
- [2014]. [Practical lessons from predicting clicks on ads at facebook](https://quinonero.net/Publications/predicting-clicks-facebook.pdf)
  - 简介：经典的GBDT+LR的解决方案，用于CTR预估，GBDT用于特征的处理，适合工业界落地的实践方案
  - 推荐指数：\* \* \* \* \*
- [2016]. [Deep Neural Networks for YouTube Recommendations](https://www.researchgate.net/publication/307573656_Deep_Neural_Networks_for_YouTube_Recommendations)
  - 简介：经典的深度学习方案在YouTube上的实践，同时包含深度学习在召回和排序过程中的应用，非常值得学习
  - 推荐指数：\* \* \* \* \*
- [2016]. [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)
  - 简介：经典的Wide & Deep网络结构，综合了记忆和泛化的能力，wide侧偏重记忆，deep侧偏重泛化；同时是一种适合工业界落地的深度学习方案
  - 推荐指数：\* \* \* \* \*
- [2016]. [Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features](https://www.kdd.org/kdd2016/subtopic/view/deep-crossing-web-scale-modeling-without-manually-crafted-combinatorial-fea)
- [2017]. [DeepFM : A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247)
  - DeepFM在Wide&Deep的基础上引入了交叉特征，使得模型能够更好的学习到组合特征
  - 推荐指数：\* \* \* \* \*
- [2017]. [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)
- [2018]. [Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba](https://arxiv.org/abs/1803.02349)
  - 简介：阿里提出的基于Graph Embedding的item的Embedding方案，相比传统的node2vec，通过增加side information解决冷启动的问题
  - 推荐指数：\* \* \* \*
  - 阅读笔记：[基于Graph Embedding的GES和EGES](http://felixzhao.cn/Articles/article/8)
- [2018]. [Real-time Personalization using Embeddings for Search Ranking at Airbnb](https://www.researchgate.net/publication/326503432_Real-time_Personalization_using_Embeddings_for_Search_Ranking_at_Airbnb)
- [2018]. [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978)
- [2018]. [Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://arxiv.org/abs/1804.07931)
  - 简介：为解决在CVR建模过程中的样本选择以及数据稀疏问题，提出ESMM（Entire Space Multi-task Model）算法，通过在全空间上直接对CVR建模，以及利用transfer learning的策略进行特征表示
  - 推荐指数：* * * * *
- [2019]. [SDM:Sequential Deep Matching Model for Online Large-scale Recommender System](https://arxiv.org/pdf/1909.00385.pdf)
  - 简介：分别对用户长期和短期兴趣建模，学习到用户的长期兴趣和短期兴趣
  - 推荐指数：* * * *
  - 阅读笔记：[序列深度匹配SDM](http://felixzhao.cn/Articles/article/11)
- [2019]. [Recommending What Video to Watch Next: A Multitask Ranking System](https://dl.acm.org/doi/10.1145/3298689.3346997)
- [2019]. [Multi-Interest Network with Dynamic Routing for Recommendation at Tmall](https://arxiv.org/abs/1904.08030)
- [2019]. [Behavior Sequence Transformer for E-commerce Recommendation in Alibaba](https://arxiv.org/abs/1905.06874v1)
  - 简介：利用Transformer对用户行为序列建模
  - 推荐指数：\* \* \* \*
  - 阅读笔记：[Transformer对用户行为序列建模算法BST](http://felixzhao.cn/Articles/article/9)
- [2019]. [Real-time Attention Based Look-alike Model for Recommender System](https://arxiv.org/abs/1906.05022)
  - 简介：实时Look-alike 算法在微信看一看中的应用
- [2019]. [Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/abs/1809.03672)
- [2019]. [Deep Learning Recommendation Model for Personalization and Recommendation Systems](https://arxiv.org/abs/1906.00091v1)
  - 简介：Facebook关于深度学习在推荐系统中落地的文章，文中提出DLRM模型，模型上重点在稀疏特征和稠密特征的处理上，同时对于如何在实践中落地提出了解决的方案
  - 推荐指数：* * * * *
- [2020]. [Controllable Multi-Interest Framework for Recommendation](https://arxiv.org/abs/2005.09347?context=cs.LG)

# 3. 内容理解

- [2014]. [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
  - 简介：CNN模型解决文本分类的问题
  - 推荐指数：\* \* \* \*
  - 阅读笔记：[CNN在文本建模中的应用TextCNN](http://felixzhao.cn/Articles/article/12)

- [2016]. [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf)
  - 简介：文本分类工具FastText的文章，FastText在当前的文本分类任务中依旧是很好的工具
  - 推荐指数：\* \* \* \*
  - 阅读笔记：[文本分类fastText算法解析](http://felixzhao.cn/Articles/article/13)

# 4. 架构工程实践

- [2020]. [为什么微信推荐这么快？](https://zhuanlan.zhihu.com/p/163189281)

# 5. 工业界解决方案

- [2017]. [【美团】旅游推荐系统的演进](https://tech.meituan.com/2017/03/24/travel-recsys.html)

- [2017]. [【美团】深度学习在美团点评推荐平台排序中的运用](https://tech.meituan.com/dl.html "深度学习在美团点评推荐平台排序中的运用")

- [【美团】美团“猜你喜欢”深度学习排序模型实践](https://tech.meituan.com/recommend_dnn.html "美团“猜你喜欢”深度学习排序模型实践")

- [【携程】推荐系统中基于深度学习的混合协同过滤模型](https://zhuanlan.zhihu.com/p/25234865 "推荐系统中基于深度学习的混合协同过滤模型")

- [2019]. [深度召回模型在QQ看点推荐中的应用实践](https://zhuanlan.zhihu.com/p/59354944)
  - 简介：介绍DeepMatch方案在QQ看点上的应用
  - 推荐指数：\* \* \*

- [2020]. [Embedding在腾讯应用宝的推荐实践](https://blog.csdn.net/Tencent_TEG/article/details/108090738)

​                                                                                                                                                                                                                                                                                                                                                                                                                                                                        



