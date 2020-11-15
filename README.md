> 欢迎大家关注个人网站**www.felixzhao.cn**，其中的部分文章会收集到该网站中。

# Recommender-System论文、学习资料以及业界分享

推荐系统（Recommender System）是机器学习应用较为成熟的方向之一，在工业界中，推荐系统也是大数据领域成功的应用之一。在一个较为完整的推荐系统中，不仅包含大家熟知的召回和排序两个阶段的常用算法之外，还会涉及到内容理解的部分的相关算法。除了算法之外，还涉及到大数据相关的处理技术以及工程实践，以下总结和整理以工业界的推荐系统为例，包含如下的几个部分：
- 推荐系统的概述及其技术架构
- 召回排序算法
- 内容理解
- 大数据处理
- 工业界解决方案

**(以下内容会持续更新)**

# 1. 推荐系统的概述及其技术架构

- [2005]. [Toward the Next Generation of Recommender Systems A Survey of the State-of-the-Art and Possible Extensions](https://ieeexplore.ieee.org/document/1423975)

  据说是推荐系统的必读文章，2005年的state-of-the-art的推荐综述，按照content-based, CF, Hybrid的分类方法进行组织，并介绍了推荐引擎设计时需要关注的特性指标，内容非常全。

- [2010]. [The YouTube video recommendation system](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Reference/The%20YouTube%20video%20recommendation%20system.pdf)

  2010年的YouTube的推荐系统的文章，文章中还没有涉及到高大上的算法，但是在文章中，值得我们借鉴的是对推荐系统的理解，包括产品的理念，数据的处理，系统的设计，是一篇值得学习的实践性的文章，建议认真研读。

- [Netflix公布个性化和推荐系统架构](http://www.infoq.com/cn/news/2013/04/netflix-ml-architecture "Netflix公布个性化和推荐系统架构")

  介绍在Netflix中的推荐系统的技术架构，对于构建工业级的推荐系统具有很重要的意义。

- [Overview of Recommender Algorithms – Part 1](https://buildingrecommenders.wordpress.com/2015/11/16/overview-of-recommender-algorithms-part-1/ "Overview of Recommender Algorithms – Part 1")

- [Overview of Recommender Algorithms – Part 2](https://buildingrecommenders.wordpress.com/2015/11/18/overview-of-recommender-algorithms-part-2/ "Overview of Recommender Algorithms – Part 2")

- [Recommender Systems in Netflix](https://buildingrecommenders.wordpress.com/2015/11/18/recommender-systems-in-netflix/ "Recommender Systems in Netflix")

- [推荐引擎初探](https://www.ibm.com/developerworks/cn/web/1103_zhaoct_recommstudy1/index.html#icomments "推荐引擎初探")

- [深度学习在推荐领域的应用](http://geek.csdn.net/news/detail/200138 "深度学习在推荐领域的应用")

- [2017]. [Deep Learning based Recommender System : A Survey and New Perspectives](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Reference/Review/Deep%20Learning%20based%20Recommender%20System%20A%20Survey%20and%20New%20Perspectives.pdf)

- [2017]. [Use of Deep Learning in Modern Recommendation System : A Summary of Recent Works](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Reference/Review/Use%20of%20Deep%20Learning%20in%20Modern%20Recommendation%20System%20A%20Summary%20of%20Recent%20Works.pdf)

- [2020]. [从零开始了解推荐系统全貌](https://zhuanlan.zhihu.com/p/259985388)
  - 简介：介绍了推荐系统的多个方面，包括用户画像，召回排序算法及策略，比较全面。
  - 推荐指数：\*\*\*

# 2. 召回排序算法

- [2014]. [Practical lessons from predicting clicks on ads at facebook](https://quinonero.net/Publications/predicting-clicks-facebook.pdf)
GBDT+LR的解决方案，用于CTR预估

- [2016]. [Wide & Deep Learning for Recommender Systems]()

- [2016]. [Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features](https://www.kdd.org/kdd2016/subtopic/view/deep-crossing-web-scale-modeling-without-manually-crafted-combinatorial-fea)

- [2017]. [DeepFM : A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247)
DeepFM

- [2017]. [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)

- [2018]. [Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba](https://arxiv.org/abs/1803.02349)
阿里提出的基于Graph Embedding的item的Embedding方案

- [2018]. [Real-time Personalization using Embeddings for Search Ranking at Airbnb]()

- [2019]. [SDM:Sequential Deep Matching Model for Online Large-scale Recommender System](https://arxiv.org/pdf/1909.00385.pdf)

- [2019]. [Recommending What Video to Watch Next: A Multitask Ranking System](https://dl.acm.org/doi/10.1145/3298689.3346997)

- [2019]. [Multi-Interest Network with Dynamic Routing for Recommendation at Tmall](https://arxiv.org/abs/1904.08030)

## 2.1. 协同过滤（collaborative filtering）

- [2000]. [Application of Dimensionality Reduction in Recommender System -- A Case Study](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Reference/CF/Application%20of%20Dimensionality%20Reduction%20in%20Recommender%20System%20--%20A%20Case%20Study.pdf)

- [2003]. [Amazon.com recommendations:Item-to-item collaborative filtering](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Reference/CF/Amazon.com%20recommendations%20Item-to-item%20collaborative%20filtering.pdf) <br />
  经典的亚马逊item-based算法的文章。

- [2005]. [A survey of collaborative filtering techniques](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Reference/CF/A%20survey%20of%20collaborative%20filtering%20techniques.pdf) <br />
  有关协同过滤的一篇综述，介绍了CF算法，所面临的一些挑战以及解决的方案，详细介绍了三种类型的CF算法：memory\_based,model\_based和hybrid，涉及到的问题以及解决方法在现今依旧值得借鉴。

- [2009]. [Matrix factorization techniques for recommender systems]()

- [2010]. [Factor in the neighbors : Scalable and accurate collaborative filtering](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Reference/CF/Factor%20in%20the%20neighbors%20Scalable%20and%20accurate%20collaborative%20filtering.pdf)

- [2016]. [Local Item-Item Models for Top-N Recommendation](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Reference/CF/Local%20Item-Item%20Models%20for%20Top-N%20Recommendation.pdf)

- [2017]. [Two Decades of Recommender Systems at Amazon.com](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Reference/CF/Two%20Decades%20of%20Recommender%20Systems%20at%20Amazon.com.pdf) <br />
  介绍在Amazon.com中使用的item-based collaborative filtering（基于项的协同过滤）算法的具体过程。

- [深入推荐引擎相关算法 - 协同过滤](https://www.ibm.com/developerworks/cn/web/1103_zhaoct_recommstudy2/index.html?ca=drs- "深入推荐引擎相关算法 - 协同过滤") <br />
  详细介绍协同过滤算法（包括基于用户的协同过滤算法和基于项的协同过滤算法）的技术细节。

## 2.2. LR

- [2007]. [Predicting clicks: estimating the click-through rate for new ads](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Reference/LR/Predicting%20clicks%20estimating%20the%20click-through%20rate%20for%20new%20ads.pdf)

## 2.3. FM

- [2010]. [Factorization Machines](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Reference/Factorization%20Machines/Factorization%20Machines.pdf)

- [2010]. [Fast Context-aware Recommendations with Factorization Machines](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Reference/Factorization%20Machines/Fast%20Context-aware%20Recommendations%20with%20Factorization%20Machines.pdf)

## 2.5. 聚类算法

- [深入推荐引擎相关算法 - 聚类](https://www.ibm.com/developerworks/cn/web/1103_zhaoct_recommstudy3/index.html?ca=drs- "深入推荐引擎相关算法 - 聚类")

## 2.6. Deep Learning

- [2007]. [Restricted Boltzmann Machines for Collaborative Filtering](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Reference/Deep%20Learning/Restricted%20Boltzmann%20Machines%20for%20Collaborative%20Filtering.pdf)

- [2013——召回]. [Learning deep structured semantic models for web search using clickthrough data](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Reference/Deep%20Learning/Learning%20deep%20structured%20semantic%20models%20for%20web%20search%20using%20clickthrough%20data.pdf)

  本文提出了DSSM模型，在原始论文中，最初是用于在搜索中计算用于是否点击的，现通常被应用在推荐中的召回阶段。

- [2015]. [A Multi-View Deep Learning Approach for Cross Domain User Modeling in Recommendation Systems](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Reference/Deep%20Learning/A%20Multi-View%20Deep%20Learning%20Approach%20for%20Cross%20Domain%20User%20Modeling%20in%20Recommendation%20Systems.pdf)

- [2015]. [AutoRec : Autoencoders Meet Collaborative Filtering](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Reference/Deep%20Learning/AutoRec%20Autoencoders%20Meet%20Collaborative%20Filtering.pdf)

  利用AutoEncoder模型学习User的隐向量矩阵U与Item的隐向量矩阵V

- [2016]. [Collaborative Denoising Auto-Encoders for Top-N Recommender Systems](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Reference/Deep%20Learning/Collaborative%20Denoising%20Auto-Encoders%20for%20Top-N%20Recommender%20Systems.pdf)

- [2016]. [Deep Neural Networks for YouTube Recommendations](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Reference/Deep%20Learning/Deep%20Neural%20Networks%20for%20YouTube%20Recommendations.pdf)

- [2016]. [Item2vec : Neural Item Embedding for Collaborative Filtering](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Reference/Deep%20Learning/Item2vec%20Neural%20Item%20Embedding%20for%20Collaborative%20Filtering.pdf) <br />



- [2016]. [Session-based recommendations with recurrent neural networks](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Reference/Deep%20Learning/Session-based%20recommendations%20with%20recurrent%20neural%20networks.pdf) <br />
  提出了利用RNN建模一个用户session间的点击序列，该方法完全利用用户在当前session里的反馈去做推荐，相比原依赖用户历史记录的推荐能在解决冷启动问题上更为简洁有效。

- [2016]. [Personal Recommendation Using Deep Recurrent Neural Networks in NetEase](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Reference/Deep%20Learning/Personal%20Recommendation%20Using%20Deep%20Recurrent%20Neural%20Networks%20in%20NetEase.pdf)

- [2016]. [Product-Based Neural Networks for User Response Prediction](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Reference/Deep%20Learning/Product-Based%20Neural%20Networks%20for%20User%20Response%20Prediction.pdf)

- [2017]. [Embedding-based News Recommendation for Millions of Users]()

- [2017]. [Deep Interest Network for Click-Through Rate Prediction](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Reference/Deep%20Learning/Deep%20Interest%20Network%20for%20Click-Through%20Rate%20Prediction.pdf)

# 3. 内容理解

- [2014]. [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
利用CNN模型解决文本分类的问题

# 4. 工业界解决方案

- [2017]. [【美团】旅游推荐系统的演进](https://tech.meituan.com/2017/03/24/travel-recsys.html)

- [【美团】深度学习在美团点评推荐平台排序中的运用](https://tech.meituan.com/dl.html "深度学习在美团点评推荐平台排序中的运用")

- [【美团】美团“猜你喜欢”深度学习排序模型实践](https://tech.meituan.com/recommend_dnn.html "美团“猜你喜欢”深度学习排序模型实践")

- [【携程】推荐系统中基于深度学习的混合协同过滤模型](https://zhuanlan.zhihu.com/p/25234865 "推荐系统中基于深度学习的混合协同过滤模型")





