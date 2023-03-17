> 欢迎大家关注个人网站[**www.felixzhao.cn**](www.felixzhao.cn)，关于下述文章的阅读笔记和一些总结都会记录在该网站中

# Recommender-System论文、学习资料以及业界分享

推荐系统（Recommender System）是大规模机器学习算法应用较为成熟的方向之一，在工业界中，推荐系统也是大数据领域成功的应用之一。在一个较为完整的推荐系统中，不仅包含大家熟知的召回和排序两个阶段的常用算法之外，对于一个完整的系统来说，还会涉及到内容理解的部分的相关算法。除了算法之外，还涉及到大数据相关的处理技术以及工程实践。

在实际的推荐系统中，通常与搜索中使用的技术互相借鉴，如下整理和总结了搜推中的一些核心技术文章，还会增加一些分析，内容主要包含如下的几个部分：

- 搜索、推荐系统综述
- 召回排序算法
- 基础模型（NLP，CV）
- 架构工程实践
- 工业界解决方案

**(以下内容会持续更新)**

# 1. 搜索、推荐系统综述
- [2005]. [Toward the Next Generation of Recommender Systems A Survey of the State-of-the-Art and Possible Extensions](https://ieeexplore.ieee.org/document/1423975)
  - 简介：据说是推荐系统的必读文章，2005年的state-of-the-art的推荐综述，按照content-based, CF, Hybrid的分类方法进行组织，并介绍了推荐引擎设计时需要关注的特性指标，内容非常全。
- [2010]. [The YouTube video recommendation system](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Reference/The%20YouTube%20video%20recommendation%20system.pdf)
  - 简介：2010年的YouTube的推荐系统的文章，文章中还没有涉及到高大上的算法，但是在文章中，值得我们借鉴的是对推荐系统的理解，包括产品的理念，数据的处理，系统的设计，初学者可以通过这篇文本对推荐系统有一个大概的认识。
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
- [2022]. [Graph Neural Networks in Recommender Systems: A Survey](https://arxiv.org/abs/2011.02260)
  - 简介：图神经网络在推荐系统中的应用

# 2. 召回、粗排、精排和重排算法以及策略
## 2.1. 召回
- [2003]. Amazon.com recommendations:Item-to-item collaborative filtering
  - 简介：亚马逊提出经典的item-based协同过滤算法，在现如今的实现中，基于i2i的算法都是召回环节的一条召回链路
- [2013]. [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf)
  - 简介：本文提出了DSSM模型，在原始论文中，最初是用于在搜索中计算用于是否点击的，现通常被应用在推荐中的相似性召回
  - 阅读笔记：[深度语义模型DSSM](http://felixzhao.cn/Articles/article/4)
- [2014]. [DeepWalk: Online Learning of Social Representations](https://arxiv.org/pdf/1403.6652.pdf)
- [2015]. [LINE: Large-scale Information Network Embedding](https://arxiv.org/pdf/1503.03578.pdf)
- [2016]. [Deep Neural Networks for YouTube Recommendations](https://dl.acm.org/doi/pdf/10.1145/2959100.2959190)
  - 简介：经典的深度学习方案在YouTube上的实践，同时包含深度学习在召回和排序过程中的应用，非常值得学习
  - 阅读笔记：[Youtube的DeepMatch模型](http://felixzhao.cn/Articles/article/15)
- [2016]. [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/pdf/1607.00653.pdf)
  - 简介：节点向量表示的一种方法，在推荐系统中的每一个item，都可以通过用户行为转换成图的表示，通过node2vec的方法，学习到图中每一个节点的向量表示，从而能够通过向量的距离度量不同节点之间的相似度
- [2017]. [Item2Vec-Neural Item Embedding for Collaborative Filtering](https://arxiv.org/pdf/1603.04259.pdf)
- [2018]. Learning Tree-based Deep Model for Recommender Systems
  - 简介：基于向量内积的召回方式计算量较大，为解决计算量的问题，文中提出**TDM**模型，用树模型构建用户兴趣
- [2018]. [Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba](https://arxiv.org/abs/1803.02349)
  - 简介：阿里提出的基于Graph Embedding的item的Embedding方案，相比传统的node2vec，通过增加side information解决冷启动的问题
  - 阅读笔记：[基于Graph Embedding的GES和EGES](http://felixzhao.cn/Articles/article/8)
- [2019]. [MOBIUS: Towards the Next Generation of Query-Ad Matching in Baidu's Sponsored Search](http://research.baidu.com/Public/uploads/5d12eca098d40.pdf)
  - 简介：对于搜索中的召回来说，召回的相关性以及召回的效率（如点击率或者转化率）都重要，文中提出在以效率为目标的前提下训练召回模型，为防止相关性漂移，在训练的过程中以相关性作为teacher进行active learning
- [2019]. [Deep Semantic Matching for Amazon Product Search](https://wsdm2019-dapa.github.io/slides/05-YiweiSong.pdf)
- [2019]. [SDM:Sequential Deep Matching Model for Online Large-scale Recommender System](https://arxiv.org/pdf/1909.00385.pdf)
  - 简介：分别对用户长期和短期兴趣建模，学习到用户的长期兴趣和短期兴趣
  - 阅读笔记：[序列深度匹配SDM](http://felixzhao.cn/Articles/article/11)
- [2019]. [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://dl.acm.org/doi/epdf/10.1145/3298689.3346996)
  - 简介：关于双塔召回中负样本的选择，文中提出在Batch内按照一定的采样概率对负样本采样
- [2019]. [Multi-interest network with dynamic routing for recommendation at Tmall](https://dl.acm.org/doi/pdf/10.1145/3357384.3357814)
  - 简介：推荐的用户多兴趣召回，因为用户的兴趣是多个维度的，在文中采用动态路由的方式生成多个用户兴趣向量，在训练时利用Label-Aware Attention机制帮助兴趣向量的训练
  - 阅读笔记：[用户多兴趣建模MIND](http://felixzhao.cn/Articles/article/51)
- [2020]. [A Comparison of Supervised Learning to Match Methods for Product Search](https://arxiv.org/abs/2007.10296)
  - 简介：商品搜索中多种Learning to Match方法的对比
- [2020]. [Beyond Lexical:A semantic Retrieval Framework for Textual Search Engine](https://arxiv.org/abs/2008.03917)
  - 简介：【知乎】关于向量召回在搜索中的应用，这样直接使用BERT作为文本语义模型并不见得会有好的效果
- [2020]. [Towards Personalized and Semantic Retrieval: An End-to-End Solution for E-commerce Search via Embedding Learning](http://arxiv.org/abs/2006.02282)
  - 简介：京东电商搜索关于个性化语义召回的文章，提出**DPSR**模型，用于解决两个方面的问题，第一是语义相关但非词匹配的召回问题，第二是个性化的召回
- [2020]. [Deep Retrieval: An End-to-End Learnable Structure Model for Large-Scale Recommendations](https://arxiv.org/pdf/2007.07203v1.pdf)
  - 简介：字节提出在双塔这种两阶段的召回过程中是有损失的，在文中提出使用DNN实现召回
- [2020]. [Embedding-based Retrieval in Facebook Search](https://arxiv.org/abs/2006.11632)
  - 简介：Facebook搜索中充分利用用户的上下文信息训练向量召回模型为用户提供相关的结果
  - 阅读笔记：[Facebook搜索的向量搜索](http://120.53.236.240/Articles/article/34)
- [2020]. [Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations](https://dl.acm.org/doi/fullHtml/10.1145/3366424.3386195)
  - 简介：对双塔结构的召回模型中的负样本分析，文中提出了一种混合采样的策略，Mixed Negative Sampling，即负样本既包括batch内采样，即相当于做unigram分布采样，另外也会对全部物料做均匀分布的随机采样，同时混合两种样本。
- [2021]. [Que2Search: Fast and Accurate Query and Document Understanding for Search at Facebook](https://dl.acm.org/doi/abs/10.1145/3447548.3467127?casa_token=_lzmp_XyNlYAAAAA%3AGOrfD2dLCe30ucy1vkacYK85-5i2agy4oKoybzluibUck-JA56hxpIGvgBa_hchJTe9fE6Dx1-3-oGs)
  - 简介：【Facebook】的一篇基于搜索中query和商品理解的文章，具有很强的实践性，文中介绍如何采用多任务和多模态的模型对query以及商品的表示，并涉及到query和商品理解的系统Que2Search的训练以及部署
- [2021]. [Embedding-based Product Retrieval in Taobao Search](http://www.researchgate.net/publication/351902997_Embedding_based_Product_Retrieval_in_Taobao_Search)
  - 简介：在搜索召回过程中，较为重要的一个问题是如何保证召回的相关性，即召回的结果与查询的query之间是相关的。在文章提出**Multi-Grained Deep Semantic Product Retrieval**（MGDSPR）算法，从两个方面，分别解决两个问题：第一，训练和推断之间的不一致；第二，如何保证相关性。
- [2021]. [Pre-trained Language Model for Web-scale Retrieval in Baidu Search](https://arxiv.org/pdf/2106.03373.pdf)
- [2021]. [Distillation based Multi-task Learning: A Candidate Generation Model for Improving Reading Duration](https://arxiv.org/pdf/2102.07142.pdf)
  - 简介：对召回的建模，首先，该召回模型是一个多目标的召回模型，两个目标分别为点击率和阅读时长，这两个目标之间存在一个先后次序，先有点击才有阅读时长，这点类似于ESMM中对CTR，CVR的建模。其次，召回模型使用的是双塔结构，双塔结构的建模通常是单目标的，因此使用蒸馏的方式，将两个目标合在一起对双塔召回蒸馏。文中设计了a distillation based multi-task learning approach，简称DMTL。（注意：这里并不是用精排模型蒸馏召回模型）
- [2021]. [A Dual Augmented Two-tower Model for Online Large-scale Recommendation](https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_4.pdf)
  - 简介：在双塔召回中存在的问题有：第一，与排序模型不一样，在双塔模型底层缺乏信息的交互，事实证明交互信息对于模型效果有着很重要的作用；第二，双塔模型中也存在着类目数据的不平衡。在文章提出了Dual Augmented Two-tower Model（DAT）模型，力图从两个方面解决上述的问题，第一，引入增强向量，同时设计了一种自适应模拟机制AMM（Adaptive-Mimic Mechanism）来学习增强向量；第二，提出了类别对齐损失CAL（Category Alignment Loss）。
- [2022]. [Adaptive Domain Interest Network for Multi-domain Recommendation](https://arxiv.org/pdf/2206.09672.pdf)
  - 简介：这是一篇介绍在召回环节考虑多场景建模的文章，在文中提出**ADI**（Adaptive Domain Interest network）网络模型，在该模型中，通过shared networks和domain-specific networks学习到不同场景间的联系与差别，整体的框架还是一个双塔的框架。

## 2.2. 粗排
- [2020]. [COLD: Towards the Next Generation of Pre-Ranking System](https://arxiv.org/abs/2007.16122v1)
  - 简介：长期以来，粗排（pre-ranking）一直被认为是精排（ranking）的简化版本，这就导致系统会陷入局部最优，文中提出COLD同时优化粗排模型和计算效率。

## 2.3. 精排
### 2.3.1. 建模方法
- [2007]. Predicting clicks: estimating the click-through rate for new ads
  - 简介：**LR**算法应用于求解CTR问题
- [2010]. Factorization Machines
  - 简介：**FM**算法在CTR中的应用
- [2010]. Fast Context-aware Recommendations with Factorization Machines
- [2013]. Ad Click Prediction: a View from the Trenches
  - 简介：针对CTR预估中的大规模计算问题，提出**FTRL在线优化算法**，算法的理论性较强，同时是非常适合实际落地的方案。
- [2014]. [Practical lessons from predicting clicks on ads at facebook](https://quinonero.net/Publications/predicting-clicks-facebook.pdf)
  - 简介：经典的**GBDT+LR**的解决方案，用于CTR预估，GBDT用于特征的处理，适合工业界落地的实践方案。
- [2016]. [Deep Neural Networks for YouTube Recommendations](https://www.researchgate.net/publication/307573656_Deep_Neural_Networks_for_YouTube_Recommendations)
  - 简介：经典的深度学习方案在YouTube上的实践，同时包含深度学习在召回和排序过程中的应用，非常值得学习
- [2016]. [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)
  - 简介：经典的**Wide & Deep**网络结构，综合了记忆和泛化的能力，wide侧偏重记忆，deep侧偏重泛化；同时是一种适合工业界落地的深度学习方案
  - 阅读笔记：[Wide & Deep算法](http://felixzhao.cn/Articles/article/18)
- [2016]. [Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features](https://www.kdd.org/kdd2016/subtopic/view/deep-crossing-web-scale-modeling-without-manually-crafted-combinatorial-fea)
  - 简介：较为经典的深度学习在CTR问题上的应用，提出**Deep Crossing**网络结构，在网络结构上与传统的MLP没有特别大的差别，唯一的区别是在MLP的计算中增加了残差的计算
  - 阅读笔记：[Deep Crossing](http://felixzhao.cn/Articles/article/23)
- [2016]. [Product-based Neural Networks for User Response Prediction](https://arxiv.org/abs/1611.00144)
  - 简介：为解决DNN不能有效处理高维特征，且方便处理交叉特征，文章中提出**PNN**网络结构，利用embedding层处理高维特征，增加product层处理特征交叉
  - 阅读笔记：[PNN网络（Product-based Neural Network）](http://felixzhao.cn/Articles/article/22)
- [2016]. [Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/pdf/1511.06939.pdf)
  - 简介：通过GRU模型对用户的历史Session建模，属于对用户序列化建模的早期文章
  - 阅读笔记：[基于Session的推荐](http://felixzhao.cn/Articles/article/10)
- [2016]. Xgboost: A scalable tree boosting system
  - 简介：这是一篇介绍Xgboost这个工具的文章，Xgboost是CTR预估中使用的一个工具，其底层原理是GBDT算法，得益于GBDT+LR的组合，使得GBDT在CTR预估领域有着广泛应用
- [2017]. [DeepFM : A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247)
  - 简介：**DeepFM**在Wide & Deep的基础上引入了交叉特征，使得模型能够更好的学习到组合特征
  - 阅读笔记：[DeepFM](http://felixzhao.cn/Articles/article/26)
- [2017]. [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)
  - 简介：对Wide & Deep模型优化，将Wide & Deep模型中的Wide部分替换成Cross network，提出了**Deep & Cross Network**，以实现特征交叉的自动化
  - 阅读笔记：[Deep&Cross Network（DCN）](http://felixzhao.cn/Articles/article/25)
- [2017]. Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction
- [2017]. [Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](https://arxiv.org/abs/1708.04617)
  - 简介：
- [2018]. [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/abs/1803.05170)
- [2018]. [Real-time Personalization using Embeddings for Search Ranking at Airbnb](https://www.researchgate.net/publication/326503432_Real-time_Personalization_using_Embeddings_for_Search_Ranking_at_Airbnb)
- [2018]. [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978)
  - 简介：根据用户的历史行为学习到的用于兴趣向量不再是固定不变的，文中提出**DIN**模型，利用local activation unit基于不同的候选学习出不同的用户兴趣
  - 阅读笔记：[深度兴趣网络DIN](http://120.53.236.240/Articles/article/32)
- [2018]. [Explainable Recommendation via Multi-Task Learning in Opinionated Text Data](https://arxiv.org/pdf/1806.03568.pdf)
- [2018]. [TEM: Tree-enhanced Embedding Model for Explainable Recommendation](http://staff.ustc.edu.cn/~hexn/papers/www18-tem.pdf)
- [2018]. [Neural Attentional Rating Regression with Review-level Explanations](http://www.thuir.cn/group/~YQLiu/publications/WWW2018_CC.pdf)
- [2018]. Self-Attentive Sequential Recommendation
- [2019]. [Order-aware Embedding Neural Network for CTR Prediction](https://dl.acm.org/citation.cfm?id=3331332)
- [2019]. [Behavior Sequence Transformer for E-commerce Recommendation in Alibaba](https://arxiv.org/abs/1905.06874v1)
  - 简介：利用Transformer对用户行为序列建模
  - 阅读笔记：[Transformer对用户行为序列建模算法BST](http://felixzhao.cn/Articles/article/9)
- [2019]. [Real-time Attention Based Look-alike Model for Recommender System](https://arxiv.org/abs/1906.05022)
  - 简介：实时Look-alike 算法在微信看一看中的应用
- [2019]. [Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/abs/1809.03672)
  - 简介：用户行为不再是孤立的，而是存在时序的关系，在DIN模型的基础上，文中提出**DIEN**模型，分别设计了兴趣抽取层和兴趣演化层对用户行为的时序关系建模
- [2019]. [Deep Learning Recommendation Model for Personalization and Recommendation Systems](https://arxiv.org/abs/1906.00091v1)
  - 简介：Facebook关于深度学习在推荐系统中落地的文章，文中提出**DLRM**模型，模型上重点在稀疏特征和稠密特征的处理上，同时对于如何在实践中落地提出了解决的方案
- [2019]. [Deep Session Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.06482.pdf)
- [2019]. [Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1904.04447.pdf)
- [2019]. [Interaction-aware Factorization Machines for Recommender Systems](https://arxiv.org/pdf/1902.09757.pdf)
- [2019]. [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)
  - 简介：通常处理特征交叉是通过Hadamard product和inner product，很少关注交叉特征的重要性，在FiBiNET中，改进特征的交叉方式以及增加特征重要行的学习，分别通过**SENET**机制动态学习特征的重要性，通过**bilinear**函数学习特征的交叉
- [2019]. [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/pdf/1810.11921.pdf)
- [2019]. [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/pdf/1904.06690.pdf)
- [2019]. [Practice on Long Sequential User Behavior Modeling for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09248.pdf)
  - 简介：用户兴趣的挖掘能够显著提升CTR预估，但随着用户行为序列的变长，一方面带来了线上的latency和storage cost上涨，同时随着长度变长，相应的模型需要相应的优化。针对上述两个方面，文中提出两个方面的优化：第一，工程方面，主要针对线上serving system的设计优化，将资源消耗最大的用户兴趣建模功能单独解构出来，设计成一个单独的模块UIC（User Interest Center）。同时，UIC维护了一个用户最新的兴趣信息，是实时更新的。第二，算法方面，提出基于memory network的**MIMN**（Multi-channel user Interest Memory Network）算法模型。
- [2020]. [Controllable Multi-Interest Framework for Recommendation](https://arxiv.org/abs/2005.09347)
- [2020]. [FuxiCTR: An Open Benchmark for Click-Through Rate Prediction](https://arxiv.org/abs/2009.05794)
  - 简介：【综述】对多种CTR模型的对比，包括浅层模型和深度模型，浅层模型包括LR，FM，FFM等，深度模型包括DNN，Wide&Deep，PNN等
- [2020]. Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction
- [2020]. [Category-Specific CNN for Visual-aware CTR Prediction at JD. com](https://dl.acm.org/doi/pdf/10.1145/3394486.3403319)
- [2020]. [Privileged Features Distillation at Taobao Recommendations](https://arxiv.org/pdf/1907.05171.pdf)
  - 简介：在实际的工作中经常需要处理训练和预测时特征不一致的问题，比如某些特征，在训练的时候能够获得，但是在预测的时候不便获得，且该特征对于模型有很正向的作用，文中提出使用蒸馏的方式，即：privileged features distillation（PFD），在Teacher模型中包含了这类privileged features，但在student模型中不再包含这类特征。
- [2020]. [Deep Match to Rank Model for Personalized Click-Through Rate Prediction](https://ojs.aaai.org//index.php/AAAI/article/view/5346)
  - 简介：在CTR建模中考虑CF的思想，提出**DMR**（Deep Match to Rank）模型，从而实现对user和item之间的相关性建模。为实现对相关性建模，在网络中引入了User-to-Item子网络和Item-to-Item子网络。
- [2021]. Self-Supervised Learning on Users' Spontaneous Behaviors for Multi-Scenario Ranking in E-commerce
- [2021]. (CIKM2021 Best Paper)[SAR-Net: A Scenario-Aware Ranking Network for Personalized Fair Recommendation in Hundreds of Travel Scenarios](https://arxiv.org/abs/2110.06475)
- [2022]. (CIKM2022 Best Paper)[Real-time Short Video Recommendation on Mobile Devices](https://arxiv.org/pdf/2208.09577.pdf)
- [2022]. [AdaSparse: Learning Adaptively Sparse Structures for Multi-Domain Click-Through Rate Prediction](https://arxiv.org/pdf/2206.13108.pdf)
- [2023]. [Decision-Making Context Interaction Network for Click-Through Rate Prediction](https://arxiv.org/pdf/2301.12402.pdf)
  - 简介：在CTR建模过程中对用户历史行为的建模，通常也只是考虑历史互动过的item，而未考虑这些item的上下文环境，文中提出**DCIN**（Decision-Making Context Interaction Network）网络，该网络中融合了**CIU**（Context Interaction Unit）以及**AIAU**（Adaptive Interest Aggregation Unit） 

### 2.3.1. position-bias
- [2019]. [Recommending What Video to Watch Next: A Multitask Ranking System](https://dl.acm.org/doi/10.1145/3298689.3346997)
  - 简介：针对样本选择时的position-bias问题，文章中提出额外使用shallow tower对position建模
- [2019]. [PAL: A Position-bias Aware Learning Framework for CTR Prediction in Live Recommender Systems](https://www.researchgate.net/publication/335771749_PAL_a_position-bias_aware_learning_framework_for_CTR_prediction_in_live_recommender_systems)
- [2020]. Bias and Debias in Recommender System: A Survey and Future Directions
- [2021]. [Deep Position-wise Interaction Network for CTR Prediction](https://arxiv.org/pdf/2106.05482.pdf)
  - 简介：

### 2.3.2. 多任务建模
- [2018]. [Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://arxiv.org/pdf/1804.07931.pdf)
  - 简介：为解决在CVR建模过程中的样本选择以及数据稀疏问题，提出**ESMM**（Entire Space Multi-task Model）算法，通过在全空间上直接对CVR建模，以及利用transfer learning的策略进行特征表示
- [2018]. [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007)
  - 简介：多任务模型的预测效果通常与任务之间的关系有关，文中提出**MMoE模型**，通过对任务之间的关系建模以达到多个目标函数以及任务之间关系的平衡
  - 阅读笔记：[Multi-gate Mixture-of-Experts（MMoE）](http://felixzhao.cn/Articles/article/30)
- [2019]. [Recommending What Video to Watch Next: A Multitask Ranking System](https://dl.acm.org/doi/10.1145/3298689.3346997)
  - 简介：文章针对两个方面的问题提出针对性的解决方案：第一个就是多目标建模问题。文章中提出MMoE（Multi-gate Mixture-of-Experts）对多目标建模。
- [2019]. [A Pareto-Efficient Algorithm for Multiple Objective Optimization in E-Commerce Recommendation](http://ofey.me/papers/Pareto.pdf)
- [2020]. [Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations](https://dl.acm.org/doi/10.1145/3383313.3412236)
  - 简介：多任务学习算法**Progressive Layered Extraction (PLE)**，是MMoE多任务学习模型的升级版本，
- [2020]. Deep Multifaceted Transformers for Multi-objective Ranking in Large-Scale E-commerce Recommender Systems

### 2.3.3. 多场景建模
- [2020]. [Improving Multi-Scenario Learning to Rank in E-commerce by Exploiting Task Relationships in the Label Space](https://cs.nju.edu.cn/_upload/tpl/01/0c/268/template268/pdf/CIKM-2020-Li.pdf)
  - 简介：文中针对多场景建模提出**HMoE**（Hybrid of implicit and explicit Mixture-of-Experts）算法，该算法类似于多任务建模中的MMoE算法，通过多个Expert结合gate网络实现对多场景的建模
- [2021]. [One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction](https://arxiv.org/abs/2101.11427)
  - 简介：文中提出**STAR**（the Star Topology Adaptive Recommender）模型，该模型包括两个部分，一部分是多场景共有的（centered network），另一部分是场景独有的（the domain-specific network）
- [2021]. [SAR-Net: A Scenario-Aware Ranking Network for Personalized Fair Recommendation in Hundreds of Travel Scenarios](https://arxiv.org/pdf/2110.06475.pdf)
- [2021]. [Personalized Transfer of User Preferences for Cross-domain Recommendation](https://arxiv.org/abs/2110.11154)
- [2022]. [Leaving No One Behind: A Multi-Scenario Multi-Task Meta Learning Approach for Advertiser Modeling](https://arxiv.org/pdf/2201.06814.pdf)
  - 简介：针对多场景对目标任务，文中提出**M2M**（multi-scenario multi-task meta learning）方法，在M2M中有三个重要部分，分别为meta unit（学习到不同场景间的关联），a meta attention module（获取到不同场景间的差异），a meta tower module（强化不同场景特征的表达能力） 
- [2022]. [APG: Adaptive Parameter Generation Network for Click-Through Rate Prediction](https://arxiv.org/abs/2203.16218)
- [2023]. [PEPNet: Parameter and Embedding Personalized Network for Infusing with Personalized Prior Information](https://arxiv.org/pdf/2302.01115.pdf)

### 2.3.4. CVR的延迟建模
- [2014]. [Modeling Delayed Feedback in Display Advertising](http://www0.cs.ucl.ac.uk/staff/w.zhang/rtb-papers/delayed-feedback.pdf)
- [2018]. [A Nonparametric Delayed Feedback Model for Conversion Rate Prediction](https://arxiv.org/abs/1802.00255)
- [2019]. [Addressing Delayed Feedback for Continuous Training with Neural Networks in CTR prediction](https://arxiv.org/abs/1907.06558)

## 2.4. 重排
- [2018]. [Learning a Deep Listwise Context Model for Ranking Refinement](https://arxiv.org/pdf/1804.05936.pdf)
- [2019]. [Personalized Re-ranking for Recommendation](https://arxiv.org/pdf/1904.06813.pdf)
  - 简介：重排
- [2022]. [Multi-Level Interaction Reranking with User Behavior History](https://arxiv.org/pdf/2204.09370.pdf)
- [2022]. [Neural Re-ranking in Multi-stage Recommender Systems: A Review](https://arxiv.org/pdf/2202.06602.pdf)
  - 简介：重排的综述
- [2022]. [Scope-aware Re-ranking with Gated Attention in Feed]()

## 2.5. 推荐多样性
- [2018]. [Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity](https://arxiv.org/pdf/1709.05135.pdf)
  - 简介：文中提出将最大化多样性问题转化为一个点行列式过程**DPP**（Determinantal Point Process），然后提出一种快速贪心算法来求解，从而降低时间复杂度
- [2019]. [DPPNET: Approximating Determinantal Point Processes with Deep Networks](https://proceedings.neurips.cc/paper/2019/file/f2d887e01a80e813d9080038decbbabb-Paper.pdf)
  - 简介：尝试使用DNN模拟DPP的输出
- [2020]. [Personalized Re-ranking for Improving Diversity in Live Recommender Systems](https://arxiv.org/pdf/2004.06390.pdf)

## 2.6. 性能优化
- [2023]. [Adaptive Low-Precision Training for Embeddings in Click-Through Rate Prediction](https://arxiv.org/pdf/2212.05735.pdf)
  - 简介：在现如今的CTR模型建模过程中，模型越来越大，这也导致了存储与推理效率两个方面的问题，在模型中，embedding table占据的空间越来越大，文章围绕如何在训练的过程中对embedding table进行压缩提出了**ALPT**（adaptive low-precision training）方法，试图在训练阶段压缩embedding table

# 3. 基础模型
[Base_Model](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Base_Model.md)

# 4. 架构工程实践
- [2020]. [为什么微信推荐这么快？](https://zhuanlan.zhihu.com/p/163189281)

# 5. 其他方向
## 5.1. 图像搜索
- [2017]. [Visual Search at eBay](https://arxiv.org/pdf/1706.03154.pdf)
- [2017]. [Visual Search at Pinterest](https://arxiv.org/pdf/1505.07647.pdf)
- [2018]. [Visual Search at Alibaba](https://arxiv.org/pdf/2102.04674.pdf)

## 5.2. Query理解

## 5.3. Query推荐
- [2018]. [RIN: Reformulation Inference Network for Context-AwareQuery Suggestion](https://jyunyu.csie.org/docs/pubs/cikm2018paper.pdf)
- [2021]. [Self-Supervised Learning on Users' Spontaneous Behaviors for Multi-Scenario Ranking in E-commerce](https://www.researchgate.net/publication/356247829_Self-Supervised_Learning_on_Users'_Spontaneous_Behaviors_for_Multi-Scenario_Ranking_in_E-commerce)

## 5.4. 向量的近似近邻检索
- [2011]. [product quantization for nearest neighbor search](https://www.researchgate.net/publication/47815472_Product_Quantization_for_Nearest_Neighbor_Search/link/00b4953c9a4b399203000000/download)
  - 简介：乘积量化**PQ**（product quantization）是向量相似检索中的一种常用算法，在Faiss中也有具体的实现，具体可参见[**IndexPQ.h**](https://github.com/facebookresearch/faiss/blob/main/faiss/IndexPQ.h)

## 5.5. 分布式训练
- [2014]. [Scaling Distributed Machine Learning with the Parameter Server](https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf)
  - 简介：著名的参数服务器**PS**


# 6. 工业界解决方案
[Industrial_Solutions](https://github.com/zhaozhiyong19890102/Recommender-System/blob/master/Industrial_Solutions.md)