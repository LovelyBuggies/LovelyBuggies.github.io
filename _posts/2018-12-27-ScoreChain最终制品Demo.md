---
layout:     post
title:      ScoreChain最终制品Demo
subtitle:   
date:       2018-12-27
author:     Nino Lau
header-img: img/post-bg-os-metro.jpg
catalog: true
tags:
    - 区块链

---



![logo](https://ws3.sinaimg.cn/large/006tNc79gy1fz3ucue48hj317q09wq3o.jpg)

经过一个月的鏖战，我人生的第一个 Dapp 做好了，取名叫 [ScoreChain](https://github.com/LovelyBuggies/Solidity_ScoreChain_Dapp) 。关于这个Dapp，您可以从 [Wiki](https://github.com/LovelyBuggies/Solidity_ScoreChain_Dapp/wiki) 获得这个项目的工程进度（~~虽然只有一个月的进度~~）。项目的使用请看 [README](https://github.com/LovelyBuggies/Solidity_ScoreChain_Dapp/blob/master/README.md)，这个文档主要是用来帮助大家熟悉这个Dapp的功能（~~当然功能很简单啦，大家都会用，主要是用来进行成果展示的啦；我很乐意分享我一个月的成果~~）。

怎样运行项目，怎样打开小狐狸 MetaMask 🦊 啥的我就不说了，[README](https://github.com/LovelyBuggies/Solidity_ScoreChain_Dapp/blob/master/README.md) 和 [Wiki](https://github.com/LovelyBuggies/Solidity_ScoreChain_Dapp/wiki) 都写的很清楚了。故事就从 localhost:3000 之后讲起。。。



### 历史遗留问题解决

上次我的[部署报告](https://github.com/LovelyBuggies/Solidity_ScoreChain_Dapp/wiki/%E9%83%A8%E7%BD%B2%E6%8A%A5%E5%91%8A)说了我目前还存在的问题：

- 一个账户只能给一个人打分；
- 打分功能还未实现。

我对这两个问题下了大功夫，翻烂了前端后端接口，终于找到了修正方案。

#### 用户多次打分

app.js 这个文件中的这段代码，规定了一个账户只能打分一次：

```js
then(function(hasSelected) {
      if(hasSelected) {
          // 只能打分一次
        $('form').hide();
      }
      loader.hide();
      content.show();
    })
```

但是一旦把hide改成了show，无限打分**叒**丧失了区中心化的透明度——TA可以无限打分。所以现在的方案是：一个账户干脆就打分一次，每个TA分配 studentNum 个打分几乎。如果TA和某个学生有不共戴天之仇，也可以对一个学生使劲打低分，但是当TA人数较多，或者每个TA被授予的打分机会比较少的时候（每个人算力相似），这个问题就不会很严重。

#### 打分功能实现

打分的实现和选学生一样，也是用框框选。之前一直想用具体数字，因为实现难度大（~~还是因为我菜鸡🐔~~）干脆就跟选学生一起用选择框算了。你们一定会质疑打分的准确性，但是事实上，这个问题根本不存在，为什么呢？

1. 同样，当TA人数较多，或者每个TA被授予的打分机会比较少的时候（每个人算力相似），这个问题就不会很严重。
2. 而且，哪个TA会真的在乎打了74还是71呢？如果说100确实和95不太一样， *我们当然也可以按着正态分布化档。*



### ScoreChain 开辟公平打分新时代

**Score Chain 的诞生是历史上开天辟地的大事，标志着学生的分数不再是模糊的了。TA们不敢再胡打分了，因为每一次打分都会被追溯，每一次希望的扼杀都会被裁决！**

现在我们来看看这个”**惊天地，泣鬼神**“的伟岸著作吧！

```shell 
npm run dev
```

调好小狐狸 🦊，现在显示界面如下。我们来看看选择框里出现了什么吧！

![](https://ws3.sinaimg.cn/large/006tNbRwly1fyic8iuuqpg30ik0nmtyk.gif)

Select Student 框框里可以选择被打分的学生，而Select Score可以选择打分的分数。现在大家还没有被打分，所以被打分次数和总分都是0，平均分NAN。我们尝试去打几个分（这里为了打分方便，我暂时设置为可以重复打分）：

![](https://ws2.sinaimg.cn/large/006tNbRwly1fyickrjm1xg30ik0nke4m.gif)

Alex 这个可怜孩子只被打了一次分，我们给他打了一个分。打完分之后，他分数变了：
![](https://ws4.sinaimg.cn/large/006tNbRwly1fyicny3g17j31d90u0aeh.jpg)

*注：我们假设这个系统在用户足够多的情况下，每个学生打分次数相似，来保证公平性。如白皮书所示，这个系统将会作为一个数据库，存放学生的作业链接。因此，这里的审计员——打分者，不仅仅可以是TA还可以是更加广泛的角色——比如学生自身。在这种方案下，一些问题（学生结盟）将会变得更多，系统的安全性将会受到挑战。*

**以太坊实在是太慢了**，趁现在网速好多打了几个分。

![](https://ws4.sinaimg.cn/large/006tNbRwly1fyicz0ddzoj319n0u0td2.jpg)

