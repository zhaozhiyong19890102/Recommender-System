> 欢迎大家关注个人网站**www.felixzhao.cn**

# Recommender-System论文、学习资料以及业界分享

推荐系统（Recommender System）是大规模机器学习算法应用较为成熟的方向之一，在工业界中，推荐系统也是大数据领域成功的应用之一。在一个较为完整的推荐系统中，不仅包含大家熟知的召回和排序两个阶段的常用算法之外，对于一个完整的系统来说，还会涉及到内容理解的部分的相关算法。除了算法之外，还涉及到大数据相关的处理技术以及工程实践。

在实际的推荐系统中，通常与搜索中使用的技术互相借鉴，如下整理和总结了搜推中的一些核心技术文章，还会增加一些分析，内容主要包含如下的几个部分：
- 搜索、推荐系统综述
- 召回排序算法
- 内容理解
- 架构工程实践
- 工业界解决方案

**(以下内容会持续更新)**

# 1. 搜索、推荐系统综述

- [2005]. [Toward the Next Generation of Recommender Systems A Survey of the State-of-the-Art and Possible Extensions](https://ieeexplore.ieee.org/document/1423975)
  - 简介：据说是推荐系统的必读文章，2005年的state-of-the-art的推荐综述，按照content-based, CF, Hybrid的分类方法进行组织，并介绍了推荐引擎设计时需要关注的特性指标，内容非常全。
- [2010]. [The YouTube video recommendation system](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Reference/The%20YouTube%20video%20recommendation%20system.pdf)
  - 简介：2010年的YouTube的推荐系统的文章，文章中还没有涉及到高大上的算法，但是在文章中，值得我们借鉴的是对推荐系统的理解，包括产品的理念，数据的处理，系统的设计，初学者可以通过这篇文本对推荐系统有一个大概的认识。
  - 推荐指数：\* \* \*
- [Netflix公布个性化和推荐系统架构](http://www.infoq.com/cn/news/2013/04/netflix-ml-architecture "Netflix公布个性化和推荐系统架构")
  - 简介：介绍在Netflix中的推荐系统的技术架构，对于构建工业级的推荐系统具有很重要的意义。
- [Overview of Recommender Algorithms – Part 1](https://buildingrecommenders.wordpress.com/2015/11/16/overview-of-recommender-algorithms-part-1/ "Overview of Recommender Algorithms – Part 1")
- [Overview of Recommender Algorithms – Part 2](https://buildingrecommenders.wordpress.com/2015/11/18/overview-of-recommender-algorithms-part-2/ "Overview of Recommender Algorithms – Part 2")
- [Recommender Systems in Netflix](https://buildingrecommenders.wordpress.com/2015/11/18/recommender-systems-in-netflix/ "Recommender Systems in Netflix")
- [推荐引擎初探](https://www.ibm.com/developerworks/cn/web/1103_zhaoct_recommstudy1/index.html#icomments "推荐引擎初探")
- [深度学习在推荐领域的应用](http://geek.csdn.net/news/detail/200138 "深度学习在推荐领域的应用")
- [2017]. [Deep Learning based Recommender System : A Survey and New Perspectives](https://arxiv.org/pdf/1707.07435.pdf)
- [2017]. [Use of Deep Learning in Modern Recommendation System : A Summary of Recent Works](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Reference/Review/Use%20of%20Deep%20Learning%20in%20Modern%20Recommendation%20System%20A%20Summary%20of%20Recent%20Works.pdf)
- [2020]. [从零开始了解推荐系统全貌](https://zhuanlan.zhihu.com/p/259985388)
  - 简介：介绍了推荐系统的多个方面，包括用户画像，召回排序算法及策略，比较全面，适合初学者了解推荐系统的全貌
  - 推荐指数：\* \* \*

# 2. 召回排序算法

## 2.1. 召回

- [2013]. [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf)
  - 简介：本文提出了DSSM模型，在原始论文中，最初是用于在搜索中计算用于是否点击的，现通常被应用在推荐中的相似性召回
  - 推荐指数：\* \* \* \* \*
  - 阅读笔记：[深度语义模型DSSM](http://felixzhao.cn/Articles/article/4)
- [2014]. [DeepWalk: Online Learning of Social Representations](https://arxiv.org/pdf/1403.6652.pdf)
- [2015]. [LINE: Large-scale Information Network Embedding](https://arxiv.org/pdf/1503.03578.pdf)
- [2016]. [Deep Neural Networks for YouTube Recommendations](https://www.researchgate.net/publication/307573656_Deep_Neural_Networks_for_YouTube_Recommendations)
  - 简介：经典的深度学习方案在YouTube上的实践，同时包含深度学习在召回和排序过程中的应用，非常值得学习
  - 推荐指数：\* \* \* \* \*
  - 阅读笔记：[Youtube的DeepMatch模型](http://felixzhao.cn/Articles/article/15)
- [2016]. [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/pdf/1607.00653.pdf)
- [2018]. Learning Tree-based Deep Model for Recommender Systems
  - 简介：基于向量内积的召回方式计算量较大，为解决计算量的问题，文中提出**TDM**模型，用树模型构建用户兴趣
  - 推荐指数：\* \* \* \*
- [2018]. [Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba](https://arxiv.org/abs/1803.02349)
  - 简介：阿里提出的基于Graph Embedding的item的Embedding方案，相比传统的node2vec，通过增加side information解决冷启动的问题
  - 推荐指数：\* \* \* \*
  - 阅读笔记：[基于Graph Embedding的GES和EGES](http://felixzhao.cn/Articles/article/8)
- [2019]. [MOBIUS: Towards the Next Generation of Query-Ad Matching in Baidu's Sponsored Search](https://dl.acm.org/doi/abs/10.1145/3292500.3330651)
  - 简介：对于搜索中的召回来说，召回的相关性以及召回的效率（如点击率或者转化率）都重要，文中提出在以效率为目标的前提下训练召回模型，为防止相关性漂移，在训练的过程中以相关性作为teacher进行active learning
- [2019]. [Deep Semantic Matching for Amazon Product Search](https://wsdm2019-dapa.github.io/slides/05-YiweiSong.pdf)
- [2019]. [SDM:Sequential Deep Matching Model for Online Large-scale Recommender System](https://arxiv.org/pdf/1909.00385.pdf)
  - 简介：分别对用户长期和短期兴趣建模，学习到用户的长期兴趣和短期兴趣
  - 推荐指数：* * * *
  - 阅读笔记：[序列深度匹配SDM](http://felixzhao.cn/Articles/article/11)
- [2020]. [A Comparison of Supervised Learning to Match Methods for Product Search](https://arxiv.org/abs/2007.10296)
  - 简介：商品搜索中多种Learning to Match方法的对比
- [2020]. [Beyond Lexical:A semantic Retrieval Framework for Textual Search Engine](https://arxiv.org/abs/2008.03917)
  - 简介：【知乎】关于向量召回在搜索中的应用
- [2020]. [Towards Personalized and Semantic Retrieval: An End-to-End Solution for E-commerce Search via Embedding Learning](http://arxiv.org/abs/2006.02282)
  - 简介：京东电商搜索关于个性化语义召回的文章，提出**DPSR**模型，用于解决两个方面的问题，第一是语义相关但非词匹配的召回问题，第二是个性化的召回
- [2020]. Deep Retrieval: An End-to-End Learnable Structure Model for Large-Scale Recommendations 
- [2020]. [Embedding-based Retrieval in Facebook Search](https://arxiv.org/abs/2006.11632)
  - 简介：Facebook搜索中充分利用用户的上下文信息训练向量召回模型为用户提供相关的结果
- [2021]. [Que2Search: Fast and Accurate Query and Document Understanding for Search at Facebook](https://dl.acm.org/doi/abs/10.1145/3447548.3467127?casa_token=_lzmp_XyNlYAAAAA%3AGOrfD2dLCe30ucy1vkacYK85-5i2agy4oKoybzluibUck-JA56hxpIGvgBa_hchJTe9fE6Dx1-3-oGs)
  - 简介：【Facebook】的一篇基于搜索中query和商品理解的文章，具有很强的实践性，文中介绍如何采用多任务和多模态的模型对query以及商品的表示，并涉及到query和商品理解的系统Que2Search的训练以及部署
- [2021]. [Embedding-based Product Retrieval in Taobao Search](http://www.researchgate.net/publication/351902997_Embedding_based_Product_Retrieval_in_Taobao_Search)
  - 简介：在搜索召回过程中，较为重要的一个问题是如何保证召回的相关性，即召回的结果与查询的query之间是相关的。在文章提出**Multi-Grained Deep Semantic Product Retrieval**（MGDSPR）算法，从两个方面，分别解决两个问题：第一，训练和推断之间的不一致；第二，如何保证相关性。
- [2021]. [Pre-trained Language Model for Web-scale Retrieval in Baidu Search](https://arxiv.org/pdf/2106.03373.pdf)

## 2.2. 排序

- [2000]. Application of Dimensionality Reduction in Recommender System -- A Case Study
- [2003]. Amazon.com recommendations:Item-to-item collaborative filtering
  - 简介：亚马逊提出经典的item-based协同过滤算法
- [2007]. Predicting clicks: estimating the click-through rate for new ads
  - 简介：LR算法应用于CTR问题
  - 推荐指数：\* \* \* \* \*
- [2010]. Factorization Machines
  - 简介：FM算法在CTR中的应用
  - 推荐指数：\* \* \* \* \*
- [2010]. Fast Context-aware Recommendations with Factorization Machines
- [2013]. Ad Click Prediction: a View from the Trenches
  - 简介：针对CTR预估中的大规模计算问题，提出**FTRL在线优化算法**，算法的理论性较强，同时是非常适合实际落地的方案
  - 推荐指数：\* \* \* \* \*
- [2014]. [Practical lessons from predicting clicks on ads at facebook](https://quinonero.net/Publications/predicting-clicks-facebook.pdf)
  - 简介：经典的**GBDT+LR**的解决方案，用于CTR预估，GBDT用于特征的处理，适合工业界落地的实践方案
  - 推荐指数：\* \* \* \* \*
- [2016]. [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653)
  - 简介：节点向量表示的一种方法，在推荐系统中的每一个item，都可以通过用户行为转换成图的表示，通过node2vec的方法，学习到图中每一个节点的向量表示，从而能够通过向量的距离度量不同节点之间的相似度
  - 推荐指数：\* \* \* \*
- [2016]. [Deep Neural Networks for YouTube Recommendations](https://www.researchgate.net/publication/307573656_Deep_Neural_Networks_for_YouTube_Recommendations)
  - 简介：经典的深度学习方案在YouTube上的实践，同时包含深度学习在召回和排序过程中的应用，非常值得学习
  - 推荐指数：\* \* \* \* \*
- [2016]. [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)
  - 简介：经典的Wide & Deep网络结构，综合了记忆和泛化的能力，wide侧偏重记忆，deep侧偏重泛化；同时是一种适合工业界落地的深度学习方案
  - 推荐指数：\* \* \* \* \*
  - 阅读笔记：[Wide & Deep算法](http://felixzhao.cn/Articles/article/18)
- [2016]. [Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features](https://www.kdd.org/kdd2016/subtopic/view/deep-crossing-web-scale-modeling-without-manually-crafted-combinatorial-fea)
  - 简介：较为经典的深度学习在CTR问题上的应用，网络结构上与传统的MLP没有特别大的差别，唯一的区别是在MLP的计算中增加了残差的计算
  - 推荐指数：\* \* \* \* \*
  - 阅读笔记：[Deep Crossing](http://felixzhao.cn/Articles/article/23)
- [2016]. [Product-based Neural Networks for User Response Prediction](https://arxiv.org/abs/1611.00144)
  - 简介：为解决DNN不能有效处理高维特征，且方便处理交叉特征，文章中提出PNN网络结构，利用embedding层处理高维特征，增加product层处理特征交叉
  - 推荐指数：\* \* \* \* \*
  - 阅读笔记：[PNN网络（Product-based Neural Network）](http://felixzhao.cn/Articles/article/22)
- [2016]. [Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/pdf/1511.06939.pdf)
  - 简介：通过GRU模型对用户的历史Session建模，属于对用户序列化建模的早期文章
  - 推荐指数：\* \* \*
  - 阅读笔记：[基于Session的推荐](http://felixzhao.cn/Articles/article/10)
- [2017]. [DeepFM : A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247)
  - 简介：DeepFM在Wide&Deep的基础上引入了交叉特征，使得模型能够更好的学习到组合特征
  - 推荐指数：\* \* \* \* \*
  - 阅读笔记：[DeepFM](http://felixzhao.cn/Articles/article/26)
- [2017]. [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)
  - 简介：对Wide & Deep模型优化，将Wide & Deep模型中的Wide部分替换成Cross network，用于自动化特征交叉
  - 推荐指数：\* \* \* \* \*
  - 阅读笔记：[Deep&Cross Network（DCN）](http://felixzhao.cn/Articles/article/25)
- [2017]. Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction
- [2017]. [Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](https://arxiv.org/abs/1708.04617)
  - 简介：
- [2017]. [Item2Vec-Neural Item Embedding for Collaborative Filtering](https://arxiv.org/pdf/1603.04259.pdf)
- [2018]. [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/abs/1803.05170)
- [2018]. [Real-time Personalization using Embeddings for Search Ranking at Airbnb](https://www.researchgate.net/publication/326503432_Real-time_Personalization_using_Embeddings_for_Search_Ranking_at_Airbnb)
- [2018]. [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978)
  - 简介：根据用户的历史行为学习到的用于兴趣向量不再是固定不变的，文中提出**DIN**模型，利用local activation unit基于不同的候选学习出不同的用户兴趣
  - 推荐指数：\* \* \*
  - 阅读笔记：[深度兴趣网络DIN](http://120.53.236.240/Articles/article/32)
- [2018]. [Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://arxiv.org/abs/1804.07931)
  - 简介：为解决在CVR建模过程中的样本选择以及数据稀疏问题，提出**ESMM**（Entire Space Multi-task Model）算法，通过在全空间上直接对CVR建模，以及利用transfer learning的策略进行特征表示
  - 推荐指数：\* \* \* \* \*
- [2018]. [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-)
  - 简介：多任务模型的预测效果通常与任务之间的关系有关，文中提出**MMoE模型**，通过对任务之间的关系建模以达到多个目标函数以及任务之间关系的平衡
  - 推荐指数：\* \* \* \* \*
  - 阅读笔记：[Multi-gate Mixture-of-Experts（MMoE）](http://felixzhao.cn/Articles/article/30)
- [2018]. [Explainable Recommendation via Multi-Task Learning in Opinionated Text Data](https://arxiv.org/pdf/1806.03568.pdf)
- [2018]. [TEM: Tree-enhanced Embedding Model for Explainable Recommendation](http://staff.ustc.edu.cn/~hexn/papers/www18-tem.pdf)
- [2018]. [Neural Attentional Rating Regression with Review-level Explanations](http://www.thuir.cn/group/~YQLiu/publications/WWW2018_CC.pdf)
- [2018]. [Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://arxiv.org/pdf/1804.07931.pdf)
- [2019]. [Order-aware Embedding Neural Network for CTR Prediction](https://dl.acm.org/citation.cfm?id=3331332)
- [2019]. [Multi-Interest Network with Dynamic Routing for Recommendation at Tmall](https://arxiv.org/abs/1904.08030)
  - 简介：聚焦在用户的兴趣的建模，不同于传统的单个兴趣向量，通过**multi-interest extractor**抽取用户的不同兴趣
  - 推荐指数：\* \* \* \* \*
- [2019]. [Behavior Sequence Transformer for E-commerce Recommendation in Alibaba](https://arxiv.org/abs/1905.06874v1)
  - 简介：利用Transformer对用户行为序列建模
  - 推荐指数：\* \* \* \*
  - 阅读笔记：[Transformer对用户行为序列建模算法BST](http://felixzhao.cn/Articles/article/9)
- [2019]. [Real-time Attention Based Look-alike Model for Recommender System](https://arxiv.org/abs/1906.05022)
  - 简介：实时Look-alike 算法在微信看一看中的应用
- [2019]. [Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/abs/1809.03672)
  - 简介：用于行为不再是孤立的，而是存在时序的关系，在DIN模型的基础上，文中提出**DIEN**模型，分别设计了兴趣抽取层和兴趣演化层对用户行为的时序关系建模
- [2019]. [Deep Learning Recommendation Model for Personalization and Recommendation Systems](https://arxiv.org/abs/1906.00091v1)
  - 简介：Facebook关于深度学习在推荐系统中落地的文章，文中提出**DLRM**模型，模型上重点在稀疏特征和稠密特征的处理上，同时对于如何在实践中落地提出了解决的方案
  - 推荐指数：\* \* \* \* \*
- [2019]. [Recommending What Video to Watch Next: A Multitask Ranking System](https://dl.acm.org/doi/10.1145/3298689.3346997)
  - 简介：多目标优化是推荐系统中一个重要的研究方向，文章为解决多目标提出Multi-gate Mixture-of-Experts，以及为解决选择偏差的问题，提出对应的解决方案
  - 推荐指数：\* \* \* \* \*
- [2019]. [Deep Session Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.06482.pdf)
- [2019]. [Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1904.04447.pdf)
- [2019]. [Personalized Re-ranking for Recommendation](https://arxiv.org/pdf/1904.06813.pdf)
- [2019]. [Interaction-aware Factorization Machines for Recommender Systems](https://arxiv.org/pdf/1902.09757.pdf)
- [2019]. [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)
- [2019]. [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/pdf/1810.11921.pdf)
- [2019]. [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/pdf/1904.06690.pdf)
- [2019]. [A Pareto-Efficient Algorithm for Multiple Objective Optimization in E-Commerce Recommendation](http://ofey.me/papers/Pareto.pdf)
- [2020]. [Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations](https://dl.acm.org/doi/10.1145/3383313.3412236)
  - 简介：多任务学习算法**Progressive Layered Extraction (PLE)**
  - 推荐指数：\* \* \* \* \*
- [2020]. [Controllable Multi-Interest Framework for Recommendation](https://arxiv.org/abs/2005.09347?context=cs.LG)
- [2020]. [FuxiCTR: An Open Benchmark for Click-Through Rate Prediction](https://arxiv.org/abs/2009.05794)
  - 简介：对多种CTR模型的对比，包括浅层模型和深度模型，浅层模型包括LR，FM，FFM等，深度模型包括DNN，Wide&Deep，PNN等
- [2020]. [COLD: Towards the Next Generation of Pre-Ranking System](https://arxiv.org/abs/2007.16122v1)
  - 简介：长期以来，粗排（pre-ranking）一直被认为是精排（ranking）的简化版本，这就导致系统会陷入局部最优，文中提出COLD同时优化粗排模型和计算效率


# 3. 内容理解

## 3.1. NLP

- [2014]. [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
  - 简介：CNN模型解决文本分类的问题
  - 推荐指数：\* \* \* \*
  - 阅读笔记：[CNN在文本建模中的应用TextCNN](http://felixzhao.cn/Articles/article/12)
- [2016]. [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf)
  - 简介：文本分类工具FastText的文章，FastText在当前的文本分类任务中依旧是很好的工具
  - 推荐指数：\* \* \* \*
  - 阅读笔记：[文本分类fastText算法解析](http://felixzhao.cn/Articles/article/13)
- [2017]. Attention Is All You Need
  - 简介：Attention的经典文章
  - 推荐指数：\* \* \* \* \*
- [2018]. Deep Contextualized Word Representations
  - 简介：预训练模型ELMo
  - 推荐指数：\* \* \* \*
  - 阅读笔记：[Embeddings from Language Models（ELMo）](http://felixzhao.cn/Articles/article/29)
- [2018]. [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
  - 简介：

## 3.2. CV

- [2017]. [Visual Search at eBay](https://arxiv.org/pdf/1706.03154.pdf)
- [2017]. [Visual Search at Pinterest](https://arxiv.org/pdf/1505.07647.pdf)
- [2018]. [Visual Search at Alibaba](https://arxiv.org/pdf/2102.04674.pdf)


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
- [2020]. [京东电商搜索中的语义检索与商品排序](https://mp.weixin.qq.com/s?__biz=MzU1NTMyOTI4Mw==&mid=2247501196&idx=1&sn=978b373e1278c709fc53036614a240ab&chksm=fbd777e0cca0fef630da6ed55518775e22afba1e74cd2e09701e055533146f29c71876ba77c4&scene=27#wechat_redirect)
- [2020]. [深度CTR预估模型在应用宝推荐系统中的探索](https://mp.weixin.qq.com/s/w9Y5yrF4gKPqNKhnMZEr-Q)
- [2021]. [万物皆为向量：在线向量召回工程服务化实践](万物皆为向量：在线向量召回工程服务化实践)
- [2021]. [爱奇艺搜索排序算法实践](https://zhuanlan.zhihu.com/p/394649477)
- [2021]. [一矢多穿：多目标排序在爱奇艺短视频推荐中的应用](https://zhuanlan.zhihu.com/p/383891318)
- [2021]. [千人万面奇妙自见：爱奇艺短视频推荐技术中多兴趣召回技术的演变](https://zhuanlan.zhihu.com/p/367305286)
- [2021]. [如何提升链路目标一致性？爱奇艺短视频推荐之粗排模型优化历程](https://zhuanlan.zhihu.com/p/353737614)
