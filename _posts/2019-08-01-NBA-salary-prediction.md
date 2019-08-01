---
layout:     post
title:     用Azure Notebook预测NBA球员薪水
subtitle:   
date:       2019-08-01
author:     Nino Lau
header-img: img/Snip20190312_61.png
catalog: true
tags:
    - 数据挖掘

---



---

## 背景介绍

> 据著名NBA记者Chris Haynes报道，在今年结束季后赛之旅后，开拓者和利拉德预计会在今年夏天达成4年1.91亿美元续约合同。💵

![](https://upload-images.jianshu.io/upload_images/3220531-cb49d4ef44b6a59f.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这份超级顶薪合同的薪水达到了令人咋舌的4260万美元、4600万美元、4940万美元和5280万美元。从0.7秒绝杀火箭🚀到独守撕裂之城；从被人质疑无缘全明星到各路打脸证明实力进入西决，利指导这份大合同可谓是货真价实。

![](https://upload-images.jianshu.io/upload_images/3220531-946e09d3a5503deb.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

但还是出现了很多质疑声，很多人认为这份合同实在是太大了。毕竟，遥想20年前，联盟的工资帽也就只有2000多万，强如乔老帮主最高才拿过3000万年薪。这还是在联盟为了收视率给了乔丹特例，同时代的巨星卡尔马龙的年薪更是只有500万，直到进入21世纪后马龙的年薪才达到了1500万。

![](https://upload-images.jianshu.io/upload_images/3220531-1ffc5f2658b84846.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

老油条们感慨生不逢时～看着如今小鲜肉们漫天的大合同，大概是哭晕在厕所了吧！😥

![](https://upload-images.jianshu.io/upload_images/3220531-31f1a35b2d485e90.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

即使是在同一年，NBA球员的贫富差距也是十分悬殊的。普遍的原则便是“能者多劳，多劳多得”。但正所谓没有对比就没有伤害：很多球星，拿着童工合同在球队当爹又当妈；而一些球员，靠着毒药合同，日常因伤休战，却依然能吃香喝辣。

![](https://upload-images.jianshu.io/upload_images/3220531-00257c315b7cff05.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

薪资分配在NBA可是关乎球队发展的大事情：钱少了，留不住超级球星；一招不慎，又有可能交着奢侈税等着来年的乐透秀。有些老板看走了眼，一激动直接断送了球队几年的未来。感慨命运不公的同时，赶紧劝NBA球队老板一句：“小老弟儿，你可长点心吧！”

![](https://upload-images.jianshu.io/upload_images/3220531-f7014765e8043697.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

那么NBA究竟是怎样"**按劳分配**"的呢？在这个实验中，我们就用Azure的Notebook工具🔧，来帮帮这群老板们把把关！

![](https://upload-images.jianshu.io/upload_images/3220531-752e6904730c1b70.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

## 项目简介

为了帮助球队老板综合评估球员实力，以及帮助NBA球员合理地定位自己的价值，此实验用[Microsoft Azure Notebook](https://notebooks.azure.com/)搭建了一些回归模型，为了初步预测NBA球员能够获得的薪水。本实验从[🏀Basketball Reference](https://www.basketball-reference.com/)爬取了2010到2019年的球员基本数据和薪资情况，经过了数据预处理之后用sklearn中多种回归方法进行预测。数据表明，这种预测虽然是有效的 ，但是在变幻莫测、瞬息万变的NBA球员市场中，仅仅依赖这种方法进行预测对老板们来说可能并不可靠。点击[**这里**](https://gallery.azure.ai/Collection/NBA-salary-prediction)查看项目的AI Gallery地址，您也可以从[**这里**](https://notebooks.azure.com/ninomyemail/projects/nba-salary-prediction)查看Azure Notebook项目原始地址获取源代码，您也可以从[**这里**](https://github.com/LovelyBuggies/NBA-Salary-Prediction)查看（*star*✨）项目的GitHub。

![](https://upload-images.jianshu.io/upload_images/3220531-60026b9fe03e07c8.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

## 使用工具

这个实验是在[Microsoft Azure Notebook](https://notebooks.azure.com/)平台上搭建了NBA球员收入的预测模型。那么为什么要用这个平台呢？[Azure Machine Learning Studio](https://studio.azureml.net/)不才是公认的[Azure](https://azure.microsoft.com/en-us/)主流机器学习开发平台么？

因为[Azure Machine Learning Studio](https://studio.azureml.net/)实在是太慢了，忒满辽......可能是因为地区的网络问题，也有可能是Studio集成了太多advanced的功能，有时候刷新个网页也需要很久🤬。很不轻量级、很不友好。举一个栗子🌰：

![](https://upload-images.jianshu.io/upload_images/3220531-f00705aa03299e65.gif?imageMogr2/auto-orient/strip)

因为我不需要用那么多功能，有个notebook就能苟延残喘了，所以我干脆大卸八块，只用了[Microsoft Azure Notebook](https://notebooks.azure.com/)平台。

打开[Azure Notebook](https://notebooks.azure.com/)，您会看到这个界面，然后[log/sign in](https://notebooks.azure.com/account/signin#)就好了（*推荐用GitHub登陆，因为这样可以直接post到您的repo中*）。

![](https://upload-images.jianshu.io/upload_images/3220531-3b72d2f0b8975070.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

谁用谁知道，用了都说好！Azure这个产品的优势是显而易见的，那就是——**简单**！进入您创建的账号，您可以看到如下页面。是不是很熟悉？是的，简直就是一个改版的“NoteHub”啊！这个平台极易上手，基本上会用[GitHub](https://github.com/)和[Jupyter Notebook](https://jupyter.org/)都能直接使用，而且项目可以直接传到您的[GitHub repo](https://github.com/)中，符合很多程序员的开发习惯。

![](https://upload-images.jianshu.io/upload_images/3220531-480336caa3fa8a4b.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

此外，[Azure Notebook](https://notebooks.azure.com/)都已经为您配置好了各种kernels，您甚至可以随意切换到R语言的kernel，简直不要太好。因为在自己本地的notebook上配置环境可能需要动到底下的conda enviroment，有时候环境一多了就完全对不上号了。而且众所周知，拥有太多conda环境很容易造成资源浪费。更厉害的是，[Azure Notebook](https://notebooks.azure.com/)竟然集成了Web开发功能，可以建立相关的Web服务。(*不过我在这个实验中暂时还没有用到，从官方[tutorial](https://docs.microsoft.com/en-us/azure/notebooks/tutorial-create-run-jupyter-notebook)上应该可以找到建立Web服务的方法。*)

最重要的是，当点了Run按钮之后，界面就变得跟本地notebook一毛一样了。这个优势对我这种配置环境树懒来说是极度友好的，因为我可以节约很多时间⌚️，把更多精力放到设计实验🧪上了！

---

## 数据预处理

可靠的回归离不开清洁的数据。

![](https://upload-images.jianshu.io/upload_images/3220531-9d3f7ca40e2a2315.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在爬取的数据集中有很多噪音：有些饮水机球员因为没得到机会上场，统计数据有缺失项；有的球员在同一年换过球队，浪迹多支球队效力；即使是在一个赛季在同一支球队效力，也会出现重复数据的情况。为了建立可用的、良好的回归模型，我们必须剔除掉一些冗余数据。

除此之外，NBA球员的赛季统计和薪水并不在同一个数据集中：球员数据放到了[league](https://www.basketball-reference.com/leagues/)目录下，而球员薪水放到了[contract](https://www.basketball-reference.com/contracts/players.html)目录下。为了将球员表现与球员薪水联系起来，我们还要尽量将两者封装到一起，以便后续处理。下面就详细地阐述一下我是怎样预处理的。

都9012年了，您首先需要更新一下您的pip，确保您的工具包没有out-of-date。


```python
import sys
!{sys.executable} -m pip install --upgrade pip
```


然后导入相关的轮子。


```python
import requests
from bs4 import BeautifulSoup
import pprint as p
import pandas as pd
import numpy as np
import re
import os
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.display.max_columns = None
```

然后我们需要建立一个data frame来存储和操作players的表现。这个data frame读取了2010到2019年每年的球员数据，并后来封装到一个pickle中。


```python

list_ = ['rank', 'player', 'pos', 'age', 'team_id', 'games', 'games_started', 'min_per_game', 'field_goals', \
         'field_goals_attempts', 'field_goals_pct', '3points', '3points_attempts', '3points_pct', '2points', \
         '2points_attempts', '2points_pct', 'efg_pct', 'free_throws', 'free_throws_attempts', 'free_throws_pct', \
         'orb', 'drb', 'trb', 'assists', 'steals', 'blocks', 'turnovers', 'personal_fouls', 'points_per_game', 'year']
player_df_final = pd.DataFrame(columns=list_)

player_df_final
```

![](https://upload-images.jianshu.io/upload_images/3220531-141e564888689611.gif?imageMogr2/auto-orient/strip)

`get_stats`是将爬取数据转化到我们data frame的函数，爬取的stats包括：`rank`，`player`，`position`，`age`，`team`，`start games`，`field goals`，`field attempts`，`field precision`，`3-points`，`3-points attemps`，`3-points precision`，`2-points`，`2-points attemps`，`2-points precision`，`free-throws`，`free-throws attemps`，`free-throws precision`，`rebounds`，`assists`，`steals`，`blocks`，`turnovers`，`fouls`以及最重要的`points per game`。


```python
def get_stats(tbody):
    player_dict = {'rank':[],'player':[], 'pos':[], 'age':[], 'team_id':[], 'games':[], 'games_started':[], 
               'min_per_game':[], 'field_goals':[], 'field_goals_attempts':[], 'field_goals_pct':[]
              ,'3points':[], '3points_attempts':[], '3points_pct':[],
              '2points':[], '2points_attempts':[], '2points_pct':[], 'efg_pct':[],
              'free_throws':[], 'free_throws_attempts':[], 'free_throws_pct':[],
              'orb':[], 'drb':[], 'trb':[],
              'assists':[], 'steals':[], 'blocks':[], 'turnovers':[], 'personal_fouls':[], 'points_per_game':[] }

    for test in tbody:
        try:

            player_dict['rank'].append(int(test.find('th', {'data-stat':'ranker'}).get_text()))
            player_dict['player'].append(test.find('td', {'data-stat':'player'}).get_text())
            player_dict['pos'].append(test.find('td', {'data-stat':'pos'}).get_text())
            player_dict['age'].append(test.find('td', {'data-stat':'age'}).get_text())
            player_dict['team_id'].append(test.find('td', {'data-stat':'team_id'}).get_text())
            player_dict['games'].append(test.find('td', {'data-stat':'g'}).get_text())
            player_dict['games_started'].append(test.find('td', {'data-stat':'gs'}).get_text())
            player_dict['min_per_game'].append(test.find('td', {'data-stat':'mp_per_g'}).get_text())
            player_dict['field_goals'].append(test.find('td', {'data-stat':'fg_per_g'}).get_text())
            player_dict['field_goals_attempts'].append(test.find('td', {'data-stat':'fga_per_g'}).get_text())
            player_dict['field_goals_pct'].append(test.find('td', {'data-stat':'fg_pct'}).get_text())
            player_dict['3points'].append(test.find('td', {'data-stat':'fg3_per_g'}).get_text())
            player_dict['3points_attempts'].append(test.find('td', {'data-stat':'fg3a_per_g'}).get_text())
            player_dict['3points_pct'].append(test.find('td', {'data-stat':'fg3_pct'}).get_text())
            player_dict['2points'].append(test.find('td', {'data-stat':'fg2_per_g'}).get_text())
            player_dict['2points_attempts'].append(test.find('td', {'data-stat':'fg2a_per_g'}).get_text())
            player_dict['2points_pct'].append(test.find('td', {'data-stat':'fg2_pct'}).get_text())
            player_dict['efg_pct'].append(test.find('td', {'data-stat':'efg_pct'}).get_text())
            player_dict['free_throws'].append(test.find('td', {'data-stat':'ft_per_g'}).get_text())
            player_dict['free_throws_attempts'].append(test.find('td', {'data-stat':'fta_per_g'}).get_text())
            player_dict['free_throws_pct'].append(test.find('td', {'data-stat':'ft_pct'}).get_text())
            player_dict['orb'].append(test.find('td', {'data-stat':'orb_per_g'}).get_text())
            player_dict['drb'].append(test.find('td', {'data-stat':'drb_per_g'}).get_text())
            player_dict['trb'].append(test.find('td', {'data-stat':'trb_per_g'}).get_text())
            player_dict['assists'].append(test.find('td', {'data-stat':'ast_per_g'}).get_text())
            player_dict['steals'].append(test.find('td', {'data-stat':'stl_per_g'}).get_text())
            player_dict['blocks'].append(test.find('td', {'data-stat':'blk_per_g'}).get_text())
            player_dict['turnovers'].append(test.find('td', {'data-stat':'tov_per_g'}).get_text())
            player_dict['personal_fouls'].append(test.find('td', {'data-stat':'pf_per_g'}).get_text())
            player_dict['points_per_game'].append(test.find('td', {'data-stat':'pts_per_g'}).get_text())

        except:
            continue


    return player_dict
```

下面我们就可以爬数据了🕷️。爬好的数据被封装到了`player_stats_total.pkl`包中。


```python
if os.path.exists('./pkl/player_stats_total.pkl') == False:
    for i in range(2010,2020):
        url = (f"https://www.basketball-reference.com/leagues/NBA_{i}_per_game.html")
        print('Scrype data from: ',url)
        response = requests.get(url)  
        html_soup = BeautifulSoup(response.content,"lxml")
        table = html_soup.find('table', {'id':'per_game_stats'})
        tbody = table.find('tbody').find_all('tr')
        player_dict = get_stats(tbody)
        player_df = pd.DataFrame(player_dict)
        player_df['year'] = i
        player_df_final = player_df_final.append(player_df, ignore_index=True)
        player_df_final.to_pickle('./pkl/player_stats_total.pkl')

    
player_df_final = pd.read_pickle('./pkl/player_stats_total.pkl')
```

爬好了球员的表现数据，我们也需要将salary考虑进去。我们引用了`nba_salaries_1990_to_2018.csv`，对17-18赛季的球员进行薪水预测。看一下里面的salary数据长什么样。


```python
salary_all_year = pd.read_csv('./csv/nba_salaries_1990_to_2018.csv')
salary_2017 = salary_all_year.query('season_start==2017')
salary_2017.head()
```

![](https://upload-images.jianshu.io/upload_images/3220531-103f8847bc8e7da7.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在这里，我们预测了latest的2017年NBA球员薪资数据，我们用`query`筛选出2017年的薪水数据。


```python
player_stat_2017 = player_df_final.query('year==2017')
player_stat_2017.head()
```

![](https://upload-images.jianshu.io/upload_images/3220531-c404cdd10e02b86b.gif?imageMogr2/auto-orient/strip)

接下来需要对数据进行`merge`来让两个数据集合并。我们先把球员名字换成小写，这样两个数据集中的数据项才可以统一地用球员名字作为key值。⚠️注意，这里不能合并两次（*在notebook上跑两次*），这样会出现多余的列。`merge`之后有很多冗余数据（比如Acy那一年就辗转多支队伍），data frame变成这样了。


```python
player_stat_2017['player'] = player_stat_2017['player'].str.replace(r'[^\w\s]', '').str.lower()
salary_2017['player'] = salary_2017['player'].str.replace(r'[^\w\s]', '').str.lower()

# note that merge can only operate once
player_stat_2017 = player_stat_2017.merge(salary_2017[['player', 'team', 'salary', ]], how = 'left', \
                                          left_on = 'player', right_on = 'player')
player_stat_2017.head()
```

![](https://upload-images.jianshu.io/upload_images/3220531-b738957b9452a1df.gif?imageMogr2/auto-orient/strip)

数据集中有些表项为null，我们剔除了这些行。所有的球员stats都`non-null`了！😊


```python
player_stat_2017 = player_stat_2017[player_stat_2017['salary'].notnull()]
player_stat_2017.info()
```

![](https://upload-images.jianshu.io/upload_images/3220531-789b5e7b9ac18ebf.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


有些球员在一个赛季辗转多支球队，这就给我们的数据处理带来很大问题。为了简化后续操作，我们选择了他们当赛季出席比赛最多的球队作为基准。

![](https://upload-images.jianshu.io/upload_images/3220531-3d0115088bad971a.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们将player和games进行升序排列，然后用`drop_duplicates`方法消除它们。这个做法可谓一石二鸟，即排除了重复的数据，同时也解决了一人多队的问题。

再来看一下Acy，只剩下一条记录了——当年比赛场次最多的篮网队。

![](https://upload-images.jianshu.io/upload_images/3220531-fa4639cb95f8810a.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


```python
player_stat_2017['games'] = player_stat_2017['games'].astype(int)
player_stat_2017.sort_values(by=['player', 'games'], ascending=[True, False], inplace=True)
player_stat_1 = player_stat_2017.drop_duplicates(subset='player', keep='first')
dup = player_stat_2017[player_stat_2017.duplicated(subset='player', keep=False)].sort_values(by='player')
rem_tot = dup.query('team_id!="TOT"')
rem_tot = rem_tot.sort_values(by=['player', 'games'], ascending=[True, False])
rem_tot = rem_tot.drop_duplicates(subset='player', keep='first')
player_stat_1 = player_stat_1.merge(rem_tot[['player', 'team_id']], how = 'left', left_on = 'player', right_on='player')
player_stat_1['team_id_y'].fillna(player_stat_1['team_id_x'], inplace= True)
player_stat_1.drop('team_id_x', axis = 1, inplace= True)
player_stat_1.rename(columns={'team_id_y':'team_id'}, inplace=True)
player_stat_1[player_stat_1['player'].str.contains('acy')]
```

![](https://upload-images.jianshu.io/upload_images/3220531-460a1defef2a0e28.gif?imageMogr2/auto-orient/strip)

另一个棘手的问题就是，随着时代发展，小球时代对传统篮球有着很强的冲击，联盟中球员的位置不再是常规的五个，联盟中出现了很多“双能卫”、“锋位摇摆人”，甚至还有可以从一打到五的全能王。

![](https://upload-images.jianshu.io/upload_images/3220531-57a30988c1977721.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这样处理起来也比较麻烦，所以我们就简单划分成“前场球员”、”后场球员“和“特殊位置”就好了。我们看一下joffry lauvergne的位置之前为大前或者出任中锋，现在变成了“特殊位置”。

![](https://upload-images.jianshu.io/upload_images/3220531-2007884f4410dd52.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


```python
player_stat_1['pos'].unique()
player_stat_1['new_pos'] = player_stat_1['pos'].apply(lambda x: 'back' if x in(['SG','PG','PG-SG']) 
                           else ('front' if x in ['PF','C','SF'] else 'special'))

player_stat_1[player_stat_1['player'].str.contains('lauvergne')]
```

![](https://upload-images.jianshu.io/upload_images/3220531-cccdf8c1de1d5e2c.gif?imageMogr2/auto-orient/strip)

我们将`team_rank.csv`也添加进去。来看一下几个詹姆斯的团队数据吧！


```python
team_rank = pd.read_csv('./csv/team_rank.csv')
player_stat_1 = player_stat_1.merge(team_rank, how = 'left', left_on='team_id', right_on='team_id')
player_stat_1.rename(columns={'rank_x':'player_rank', 'rank_y':'team_rank', 'points':'team_points'}, inplace=True)
player_stat_1[player_stat_1['player'].str.contains('james')]
```

![](https://upload-images.jianshu.io/upload_images/3220531-a472ff85d3b263af.gif?imageMogr2/auto-orient/strip)

为了便于处理数据，我们重建data frame，并指定了每一个stat的数据类型。


```python
int_col = ['age',  'games', 'games_started', 'team_rank']
float_col = ['min_per_game', 'field_goals', 'field_goals_attempts',
       'field_goals_pct', '3points', '3points_attempts', '3points_pct',
       '2points', '2points_attempts', '2points_pct', 'efg_pct', 'free_throws',
       'free_throws_attempts', 'free_throws_pct', 'orb', 'drb', 'trb',
            'assists', 'steals', 'blocks', 'turnovers', 'personal_fouls',
       'points_per_game', 'team_points']
string_col = ['team_id']

player_stat_1[int_col] = player_stat_1[int_col].apply(pd.to_numeric, axis = 1)
player_stat_1[float_col] = player_stat_1[float_col].apply(pd.to_numeric, axis = 1)
player_stat_1[string_col] = player_stat_1[string_col].astype('|S5')
player_stat_1.info()
```

![](https://upload-images.jianshu.io/upload_images/3220531-86f5e9e8887e7d27.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


这样，数据预处理就暂时完成了，我们把它封装到了`data_processing_over.pkl`包中。然后让我们看一下NBA里保罗**们**的数据吧！


```python
player_stat_1.to_pickle('./pkl/data_processing_over.pkl')
player_stat_1[player_stat_1['player'].str.contains('paul')]
```

![](https://upload-images.jianshu.io/upload_images/3220531-e43fd2930e6d35a9.gif?imageMogr2/auto-orient/strip)

---

## 分析

处理数据之前，我们首先需要分析一下数据集。

### 数据可视化

首先我们将数据集可视化了一下，可以看出：高薪球员大部分都在30岁左右，年少有为的年轻球员还是少；联盟的老将有些也虎落平阳。此外，综合来看，**前场球员拿到高薪水的多，虽然他们得分并不比后卫们优秀**。在助攻方面，后卫比较强势；在篮板方面，锋线比较厉害。

![](https://upload-images.jianshu.io/upload_images/3220531-3d40736112f81c63.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


```python
all_data = pd.read_pickle('./pkl/data_processing_over.pkl')

X_vis = all_data[['points_per_game','assists','trb','steals', 'blocks','steals','turnovers','age','new_pos','team_rank']]
y_vis = all_data[['salary']]

# PPG
fig = plt.figure(figsize=(16,8))
fig.add_subplot(311)
plt.grid(lw=0)
cm = plt.cm.get_cmap('PuBu')
for i in range(len(X_vis)):
    xs = float(X_vis['points_per_game'][i])
    zs = float(y_vis['salary'][i])
    if X_vis['new_pos'][i]=='front':
        ma = '^'
    else:
        ma = 'o' 
    plt.scatter(xs,zs,s=35,c=int(X_vis['age'][i]),vmin=18, vmax=38, cmap=cm, marker=ma, alpha=0.9)
cb=plt.colorbar()
font = {'family' : 'serif',
        'color'  : '#000000',
        'weight' : 'normal',
        }
plt.xlabel('Point Per Game')
plt.ylabel('Player Salary')
cb.set_label('Player Age',fontdict=font)
plt.show()

# Asists
fig = plt.figure(figsize=(16,8))
fig.add_subplot(312)
plt.grid(lw=0)
cm = plt.cm.get_cmap('Reds')
for i in range(len(X_vis)):
    xs = float(X_vis['assists'][i])
    zs = float(y_vis['salary'][i])
    if X_vis['new_pos'][i]=='front':
        ma = '^'
    else:
        ma = 'o' 
    plt.scatter(xs,zs,s=35,c=int(X_vis['age'][i]),vmin=18, vmax=38, cmap=cm, marker=ma, alpha=0.9)
cb=plt.colorbar()
font = {'family' : 'serif',
        'color'  : '#000000',
        'weight' : 'normal',
        }
plt.xlabel('Assists')
plt.ylabel('Player Salary')
cb.set_label('Player Age',fontdict=font)
plt.show()

# Rebounds
fig = plt.figure(figsize=(16,8))
fig.add_subplot(313)
plt.grid(lw=0)
cm = plt.cm.get_cmap('Purples')
for i in range(len(X_vis)):
    xs = float(X_vis['trb'][i])
    zs = float(y_vis['salary'][i])
    if X_vis['new_pos'][i]=='front':
        ma = '^'
    else:
        ma = 'o' 
    plt.scatter(xs,zs,s=35,c=int(X_vis['age'][i]),vmin=18, vmax=38, cmap=cm, marker=ma, alpha=0.9)
cb=plt.colorbar()
font = {'family' : 'serif',
        'color'  : '#000000',
        'weight' : 'normal',
        }
plt.xlabel('Total Rebounds')
plt.ylabel('Player Salary')
cb.set_label('Player Age',fontdict=font)
plt.show()

```


![](https://upload-images.jianshu.io/upload_images/3220531-e01e9ea178030fd4.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



![](https://upload-images.jianshu.io/upload_images/3220531-0da656456502f9eb.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



![](https://upload-images.jianshu.io/upload_images/3220531-49871b1edc81850e.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


### 相关性分析

因为很多球员的实力指标是重叠的，太多指标意味着接下来高维度的处理。为了选择一些有用的实力指标，我们需要进行相关性分析。这里我们借助`seaborn`工具将各个实力指标的相关性画了出来。


```python
all_data = pd.read_pickle('./pkl/data_processing_over.pkl')
all_data['Position_back']=np.where((all_data['new_pos']=='back'),1,0)
all_data['Position_front']=np.where((all_data['new_pos']=='front'),1,0)
tmp = all_data[['age', 'games', 'games_started', 'min_per_game', 'field_goals', 'field_goals_attempts', \
                '2points', '2points_attempts', '3points', '3points_attempts',  'free_throws', 'orb', 'drb', \
                'trb', 'assists', 'steals', 'blocks','free_throws_attempts', 'turnovers', 'personal_fouls', \
                'points_per_game', 'team_points','pos']]

if os.path.exists('./fig/sns/sns_plot.pdf') == False:
    sns_plot = sns.pairplot(data = tmp, hue='pos', diag_kind="kde", markers="+",
                              plot_kws=dict(s=50, edgecolor="b", linewidth=1),
                              diag_kws=dict(shade=True))
    sns_plot.savefig("./fig/sns/sns_plot.pdf")
```

![](https://upload-images.jianshu.io/upload_images/3220531-a6638b82f3ebf137.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

从图中，我们可以看出有一些相关性很强的指标，在之后我们就可以舍弃掉它们中的一个，这样可以达到降维的效果。比如投篮尝试次数越多，得分越高；一般只有能够得到运动战得分的球员才有资格参加比赛（*也可以说是因为他们参加了比赛所以才有运动战得分*）。还有一些相关性不强的变量，比如三分尝试次数和篮板，三分命中率和个人犯规次数等关系相关性很弱，对于这种情况，我们一般把这些指标视作独立维度用来回归。

![](https://upload-images.jianshu.io/upload_images/3220531-b7e41ed115356700.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

为了更加精确地阐述各个指标之间的关系，我们又画出来了corrcoef图。pairplot中越相关（*呈线形*）的关系在corrcoef中越“热”。对角线上表示自反关系，所以都是1.00 。


```python
tmp = all_data[['age', 'games', 'games_started', 'min_per_game', 'field_goals', 'field_goals_attempts', \
                '2points', '2points_attempts', '3points', '3points_attempts',  'free_throws', 'orb', 'drb', \
                'trb', 'assists', 'steals', 'blocks','free_throws_attempts', 'turnovers', 'personal_fouls', \
                'points_per_game', 'team_points']]

cm = np.corrcoef(tmp.values.T)   
sns.set(font_scale=.8)
plt.figure(figsize=(12,12))
sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 11},yticklabels=tmp.T,xticklabels=tmp.T)
plt.savefig('./fig/sns/sns_coeff.pdf', dpi=300)
plt.show()
```


![](https://upload-images.jianshu.io/upload_images/3220531-f07d3c791e9532e1.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


## 回归

![](https://upload-images.jianshu.io/upload_images/3220531-58a71bb66ff303c8.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

不要慌！我们这就用各种回归模型给你们算工资💰！

为了避免造轮子的麻烦，我们直接调用了[sklearn](https://scikit-learn.org/stable/)的API，并根据[黄砖头](http://www.scikit-yb.org/en/latest/)对模型评估时附带的得分对每种方法进行评估。最好的得分是1.0，当然会得到负的分数。模型分数的定义为$R^2=(1-\mu/\nu)$, where $\mu=((y_{true}-y_{predict})**2).sum()$, $\nu=((y_{true}-y_{mean})**2).sum()$。

其实我作为数据挖掘小白对我这个数据集中的数据、对各种回归模型的优劣势并没有很清晰的认识。我的principle就是**“试”**，所以我几乎尝试了所有`sklearn`中能够用到我这个数据集上的回归模型。不得不说，`sklearn`真的是超级良心，主页上的六个modules直接为我们指引了方向。只需调用您需求功能的API，不停尝试换参和换模型，就可以达到理想的结果。“它可真是数据分析世界的良心啊！”

![](https://upload-images.jianshu.io/upload_images/3220531-c171ba8d7921bc32.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

首先需要先下一个[黄砖头](http://www.scikit-yb.org/en/latest/)🧱用来可视化和初步评估回归模型的性能。


```python
import sys
!{sys.executable} -m pip install yellowbrick
```


然后把轮子都import一下。


```python
import sklearn
import yellowbrick
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, \
ExtraTreesRegressor
from yellowbrick.regressor import PredictionError
from sklearn.neural_network import MLPRegressor
```

在回归之前，我们还需要重新定义一下数据。首先用PCA再降一下维，然后分裂数据集成训练集和测试集。


```python
# Specify the features of interest and the target

# Extract the instances and target
X = all_data[['age', 'games', 'games_started', 'min_per_game', 'field_goals', 'field_goals_attempts', \
              '2points', '2points_attempts', '3points', '3points_attempts',  'free_throws', 'orb',\
              'drb', 'trb', 'assists', 'steals', 'blocks','free_throws_attempts', \
              'turnovers', 'personal_fouls', 'points_per_game', 'team_points']]
y = all_data['salary']

# PCA decomposition -- seems do not effect...
pca = decomposition.PCA(n_components=8)
pca.fit(X)
X = pca.transform(X)

# Create the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

下面我们就可以正式开始回（shi）归（can）啦！

![](https://upload-images.jianshu.io/upload_images/3220531-ca31d106bf07158b.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 广义线性模型

首先，我们尝试了一组广义线性回归的方法，其中目标值预期为特征的线性组合。

$$\hat{y}(w,x)=w_0+w_1 x_1+\cdots+w_p x_p$$

#### 多元线性回归

我们先采用最简单的线性模型——多元线性回归。线性回归是利用数理统计中回归分析，来确定两种或两种以上变量间相互依赖的定量关系的一种统计分析方法。其表达形式为$y = w'x+e$，$e$为误差服从均值为0的正态分布。

![](https://upload-images.jianshu.io/upload_images/3220531-4cd0fcf28c5a3f28.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们先定义好一个线性回归器，然后用黄转头🧱可视化并评分，最后用散点图展示了419个有效数据中球员真实工资与预测薪水之间的差距。


```python
# Instantiate the model and visualizer
lr = LinearRegression()
visualizer = PredictionError(lr)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data

# Prediction plot (simple LR using scatter)
predictions = lr.predict(X)
prd_lr = predictions.astype(int)
x = range(len(prd_lr))
plt.figure(figsize=(6,4))
plt.grid(ls=':', lw=1)
plt.scatter(x, prd_lr, color='#FEB64D', marker='o', facecolor='None', lw=2)
plt.scatter(x, y, color='#9287E7', marker='o')
label = ["Prediction", "Actual"]
plt.legend(label, loc=2, markerscale=0.85, ncol = 1, fontsize=10, framealpha=1)
plt.title('Linear Regression', fontsize=12)
plt.savefig("./fig/reg/lr.png")
plt.show()

# Show score
print("Model Score: ",lr.score(X_test, y_test))
```


![](https://upload-images.jianshu.io/upload_images/3220531-5da7c6f0cca5997d.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](https://upload-images.jianshu.io/upload_images/3220531-91b70e9c9ecabd69.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


    Model Score:  0.45663550413795606


#### Lasso回归

![](https://upload-images.jianshu.io/upload_images/3220531-ca63a9a51cf42ff8.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Lasso回归模型是一个用于估计稀疏参数的线性模型，Lasso在线性模型上加上了一个l1正则项，特别适用于参数数目缩减。基于这个原因，Lasso回归模型在压缩感知中应用的十分广泛。

![](https://upload-images.jianshu.io/upload_images/3220531-6901501e3f210645.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们先定义好一个Lasso回归器，然后用黄转头🧱可视化并评分，最后用散点图展示了419个有效数据中球员真实工资与预测薪水之间的差距。


```python
# Instantiate the model and visualizer
lasso = Lasso()
visualizer = PredictionError(lasso)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data

# Prediction plot
predictions = lasso.predict(X)
prd_lasso = predictions.astype(int)
plt.figure(figsize=(6,4))
x = range(len(prd_lasso))
plt.grid(ls=':', lw=1)
plt.scatter(x, prd_lasso, color='#FEB64D', marker='o', facecolor='None', lw=2)
plt.scatter(x, y, color='#9287E7', marker='o')
label = ["Prediction", "Actual"]
plt.legend(label, loc=2, markerscale=0.85, ncol = 1, fontsize=10, framealpha=1)

plt.title('Lasso Regression', fontsize=12)
plt.savefig("./fig/reg/lasso.png")
plt.show()

# Show score
print("Model Score: ",lasso.score(X_test, y_test))
```


![](https://upload-images.jianshu.io/upload_images/3220531-3f10b3fc562c88ad.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](https://upload-images.jianshu.io/upload_images/3220531-b524f85ead0cc9aa.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


    Model Score:  0.45663555627227415


#### 岭回归

岭回归是一种专用于共线性数据分析的有偏估计回归方法，实质上是一种改良的最小二乘估计法，通过放弃最小二乘法的无偏性，以损失部分信息、降低精度为代价获得回归系数更为符合实际、更可靠的回归方法，对病态数据的拟合要强于最小二乘法。

![](https://upload-images.jianshu.io/upload_images/3220531-7dee4b6c9c4914a3.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们先定义好一个$\alpha=0.1$的Ridge回归器，然后用黄转头🧱可视化并评分，用散点图展示了419个有效数据中球员真实工资与预测薪水之间的差距。之后我们又尝试了`alphas=[0.01, 0.1, 1.0, 10.0]`的ridgeCV回归器，这种回归器的优势在于可以自动地挑选更加合适的参数。经过尝试，发现CV回归器确实能够enhance原有的回归器。

![](https://upload-images.jianshu.io/upload_images/3220531-c929e6e0f6185e63.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


```python
# Instantiate the model and visualizer
ridge = Ridge(alpha=0.1)
visualizer = PredictionError(ridge)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data

# Prediction plot
predictions = ridge.predict(X)
prd_ridge = predictions.astype(int)
plt.figure(figsize=(6,4))
x = range(len(prd_ridge))
plt.grid(ls=':', lw=1)
plt.scatter(x, prd_ridge, color='#FEB64D', marker='o', facecolor='None', lw=2)
plt.scatter(x, y, color='#9287E7', marker='o')
label = ["Prediction", "Actual"]
plt.legend(label, loc=2, markerscale=0.85, ncol = 1, fontsize=10, framealpha=1)
plt.title('Ridge Regression', fontsize=12)
plt.savefig("./fig/reg/ridge.png")
plt.show()

# Show score
print("Model Score: ",ridge.score(X_test, y_test))
```


![](https://upload-images.jianshu.io/upload_images/3220531-c6a89b1cfa6f7410.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](https://upload-images.jianshu.io/upload_images/3220531-4d4d7337339ba06a.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


    Model Score:  0.45664410518908105



```python
# Instantiate the model and visualizer
ridgeCV = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
visualizer = PredictionError(ridgeCV)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data

# Prediction plot
predictions = ridgeCV.predict(X)
prd_ridgeCV = predictions.astype(int)
plt.figure(figsize=(6,4))
x = range(len(prd_ridgeCV))
plt.grid(ls=':', lw=1)
plt.scatter(x, prd_ridgeCV, color='#FEB64D', marker='o', facecolor='None', lw=2)
plt.scatter(x, y, color='#9287E7', marker='o')
label = ["Prediction", "Actual"]
plt.legend(label, loc=2, markerscale=0.85, ncol = 1, fontsize=10, framealpha=1)

plt.title('RidgeCV Regression', fontsize=12)
plt.savefig("./fig/reg/ridgeCV.png")
plt.show()

# Show score
print("Model Score: ",ridgeCV.score(X_test, y_test))
```


![](https://upload-images.jianshu.io/upload_images/3220531-720311865720655e.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](https://upload-images.jianshu.io/upload_images/3220531-5f443e40a0b1fc6c.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


    Model Score:  0.4574808811208244


#### 弹性网络

弹性网络是一种线性回归模型，训练了$l_1$和$l_2$-范数正则化系数。这种组合允许学习稀疏模型，其中很少的权重是非零的，同时仍然保持岭的正则化属性。利用`l1_ratio`参数控制l1和l2的凸组合。当存在多个相互关联的特征时，弹性网络可能同时选择两个。它在Lasso和Ridge之间进行权衡，允许弹性网在旋转时继承Ridge的一些稳定性。

![](https://upload-images.jianshu.io/upload_images/3220531-9f70e7b55d7235f9.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


```python
# Instantiate the model and visualizer
enet = ElasticNet(alpha=0.1, l1_ratio=0.7)
visualizer = PredictionError(enet)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data

# Prediction plot
predictions = enet.predict(X)
prd_enet = predictions.astype(int)
plt.figure(figsize=(6,4))
x = range(len(prd_enet))
plt.grid(ls=':', lw=1)
plt.scatter(x, prd_enet, color='#FEB64D', marker='o', facecolor='None', lw=2)
plt.scatter(x, y, color='#9287E7', marker='o')
label = ["Prediction", "Actual"]
plt.legend(label, loc=2, markerscale=0.85, ncol = 1, fontsize=10, framealpha=1)

plt.title('Elastic Net Regression', fontsize=12)
plt.savefig("./fig/reg/enet.png")
plt.show()

# Show score
print("Model Score: ",enet.score(X_test, y_test))
```


![](https://upload-images.jianshu.io/upload_images/3220531-563a8c78b9810327.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](https://upload-images.jianshu.io/upload_images/3220531-03747418bd5d7300.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


    Model Score:  0.4574576128442218


### 支持向量机

支持向量机分类方法可以推广到回归问题的求解，称为支持向量回归。

![](https://upload-images.jianshu.io/upload_images/3220531-d18ff1eddec2f498.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

支持向量分类生成的模型只依赖于训练数据的子集，构建模型的成本函数并不关心超出边界的训练点。支持向量回归生成的模型只依赖于训练数据的一个子集，而且用于构建模型的成本函数忽略了任何接近于模型预测的训练数据。

在这里，我们先定义好一个支持向量回归器，然后用黄转头🧱可视化并评分，最后用散点图展示了419个有效数据中球员真实工资与预测薪水之间的差距。

```python
# Instantiate the model and visualizer
svr = SVR(kernel='poly')
visualizer = PredictionError(svr)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data

# Prediction plot
predictions = svr.predict(X)
prd_svr = predictions.astype(int)
plt.figure(figsize=(6,4))
x = range(len(prd_svr))
plt.grid(ls=':', lw=1)
plt.scatter(x, prd_svr, color='#FEB64D', marker='o', facecolor='None', lw=2)
plt.scatter(x, y, color='#9287E7', marker='o')
label = ["Prediction", "Actual"]
plt.legend(label, loc=2, markerscale=0.85, ncol = 1, fontsize=10, framealpha=1)

plt.title('Support Vector Regression', fontsize=12)
plt.savefig("./fig/reg/svr.png")
plt.show()

# Show score
print("Model Score: ",svr.score(X_test, y_test))
```



![](https://upload-images.jianshu.io/upload_images/3220531-a49b91ef6af32423.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](https://upload-images.jianshu.io/upload_images/3220531-11ae8867a9e759c6.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


    Model Score:  0.3449399132513786


### 决策树

![](https://upload-images.jianshu.io/upload_images/3220531-ce9bd9df8c74777c.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

决策树是一种用于分类和回归的非参数监督学习方法。目标是创建一个模型，通过学习从数据特性推断出的简单决策规则来预测目标变量的值。例如，在下面的示例中，决策树从数据中学习如何使用一组if-then-else决策规则来近似正弦曲线。树越深，决策规则越复杂，模型越合适。

![](https://upload-images.jianshu.io/upload_images/3220531-0ed09afdd9dccf8f.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在这里，我们先定义好一个Decision Tree回归器，然后用黄转头🧱可视化并评分，最后用散点图展示了419个有效数据中球员真实工资与预测薪水之间的差距。


```python
# Instantiate the model and visualizer
decision_tree = DecisionTreeRegressor()
# decision_tree = DecisionTreeRegressor(max_depth=3)
visualizer = PredictionError(decision_tree)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data

# Prediction plot
predictions = decision_tree.predict(X)
prd_decision_tree = predictions.astype(int)
plt.figure(figsize=(6,4))
x = range(len(prd_decision_tree))
plt.grid(ls=':', lw=1)
plt.scatter(x, prd_decision_tree, color='#FEB64D', marker='o', facecolor='None', lw=2)
plt.scatter(x, y, color='#9287E7', marker='o')
label = ["Prediction", "Actual"]
plt.legend(label, loc=2, markerscale=0.85, ncol = 1, fontsize=10, framealpha=1)

plt.title('Decision Tree Regression', fontsize=12)
plt.savefig("./fig/reg/decision_tree.png")
plt.show()

# Show score
print("Model Score: ",decision_tree.score(X_test, y_test))
```


![](https://upload-images.jianshu.io/upload_images/3220531-cb9423d5b1be740c.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](https://upload-images.jianshu.io/upload_images/3220531-4da2111939d7f81d.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


    Model Score:  0.4176739276920418


### 近邻方法

最近邻方法的原理是找到距离新点最近的预定义训练样本个数，并据此预测标签。

![](https://upload-images.jianshu.io/upload_images/3220531-d1772cc4a20622f0.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

顾名思义，K-最近邻方法的原理自然就是是找到距离新点最近的预定义训练样本个数，并据此预测标签。样本的数量可以是一个用户定义的常数，也可以根据点的局部密度变化。一般来说，距离可以是任何度量：标准欧氏距离是最常见的选择。

![](https://upload-images.jianshu.io/upload_images/3220531-015125203f14d952.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在这里，我们先定义好一个KNN回归器，然后用黄转头🧱可视化并评分，最后用散点图展示了419个有效数据中球员真实工资与预测薪水之间的差距。


```python
# Instantiate the model and visualizer
knn = KNeighborsRegressor(weights="uniform")
visualizer = PredictionError(knn)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data

# Prediction plot
predictions = knn.predict(X)
prd_knn = predictions.astype(int)
plt.figure(figsize=(6,4))
x = range(len(prd_knn))
plt.grid(ls=':', lw=1)
plt.scatter(x, prd_knn, color='#FEB64D', marker='o', facecolor='None', lw=2)
plt.scatter(x, y, color='#9287E7', marker='o')
label = ["Prediction", "Actual"]
plt.legend(label, loc=2, markerscale=0.85, ncol = 1, fontsize=10, framealpha=1)

plt.title('KNN Regression', fontsize=12)
plt.savefig("./fig/reg/knn.png")
plt.show()

# Show score
print("Model Score: ",knn.score(X_test, y_test))
```


![](https://upload-images.jianshu.io/upload_images/3220531-57bc1dd77191ad5e.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](https://upload-images.jianshu.io/upload_images/3220531-4c73fb18e94c9541.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


    Model Score:  0.4736951433350763


### 集成方法

集成方法的目标是将多个基本估计器的预测与给定的学习算法相结合，从而提高对单个估计器的通用性和鲁棒性。

集成方法通常分为两大类:

1. 在**平均方法**中，驱动原则是独立地构建几个估计器，然后对它们的预测进行平均。平均而言，组合估计量通常比任何单基估计量都好，因为它的方差减小了；

2. 在**增强方法**中，基本估计量是按顺序建立的，并试图减少组合估计量的偏差。这样做的动机是将几个较弱的模型组合起来，形成一个强大的整体。

#### 随机森林

在随机森林中，集合中的每棵树都是根据从训练集中抽取的样本进行替换而构建的。此外，在构建树的过程中对每个节点进行拆分时，可以从所有输入特性或max_features大小的随机子集中找到最佳拆分。

![](https://upload-images.jianshu.io/upload_images/3220531-bfb41e9c067cd71f.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这两个随机性来源的目的是降低森林估计量的方差。单个决策树通常表现出很高的方差，并且倾向于过度拟合。森林中注入的随机性产生了具有一定解耦预测误差的决策树。通过对这些预测取平均值，一些错误可以抵消。随机森林通过组合不同的树木来减少方差，有时以略微增加偏差为代价。

![](https://upload-images.jianshu.io/upload_images/3220531-21e007c2c4a6c3ae.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在这里，我们先定义好一个随即森林🌲回归器，然后用黄转头🧱可视化并评分，最后用散点图展示了419个有效数据中球员真实工资与预测薪水之间的差距。


```python
# Instantiate the model and visualizer
rfr = RandomForestRegressor()
visualizer = PredictionError(rfr)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data

# Prediction plot
predictions = rfr.predict(X)
prd_rfr = predictions.astype(int)
plt.figure(figsize=(6,4))
x = range(len(prd_rfr))
plt.grid(ls=':', lw=1)
plt.scatter(x, prd_rfr, color='#FEB64D', marker='o', facecolor='None', lw=2)
plt.scatter(x, y, color='#9287E7', marker='o')
label = ["Prediction", "Actual"]
plt.legend(label, loc=2, markerscale=0.85, ncol = 1, fontsize=10, framealpha=1)

plt.title('Random Forest Regression', fontsize=12)
plt.savefig("./fig/reg/rfr.png")
plt.show()

# Show score
print("Model Score: ",rfr.score(X_test, y_test))
```



![](https://upload-images.jianshu.io/upload_images/3220531-7fbf03a287cb8fb7.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](https://upload-images.jianshu.io/upload_images/3220531-cdfc2ff770a38b76.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


    Model Score:  0.5179059365052248


#### 非常随机树

在Extremely Randomized Trees中，随机性在分割计算的方式上更进一步。与随机森林一样，使用候选特征的随机子集，但不是寻找最具鉴别性的阈值，而是为每个候选特征随机绘制阈值，并选取这些随机生成的阈值中最好的作为分割规则。这通常允许稍微减少模型的方差，但代价是偏移的稍微增加。

在这里，我们先定义好一个Extra Trees回归器，然后用黄转头🧱可视化并评分，最后用散点图展示了419个有效数据中球员真实工资与预测薪水之间的差距。


```python
# Instantiate the model and visualizer
etr = ExtraTreesRegressor()
visualizer = PredictionError(etr)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data

# Prediction plot
predictions = etr.predict(X)
prd_etr = predictions.astype(int)
plt.figure(figsize=(6,4))
plt.grid(ls='--', lw=2)
x = range(len(prd_etr))
plt.grid(ls=':', lw=1)
plt.scatter(x, prd_etr, color='#FEB64D', marker='o', facecolor='None', lw=2)
plt.scatter(x, y, color='#9287E7', marker='o')
label = ["Prediction", "Actual"]
plt.legend(label, loc=2, markerscale=0.85, ncol = 1, fontsize=10, framealpha=1)

plt.title('Extra Trees Regression', fontsize=12)
plt.savefig("./fig/reg/etr.png")
plt.show()

# Show score
print("Model Score: ",etr.score(X_test, y_test))
```



![](https://upload-images.jianshu.io/upload_images/3220531-41ac8db4e2d3fdce.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](https://upload-images.jianshu.io/upload_images/3220531-36f29e7c83accfa4.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


    Model Score:  0.5157097221777236


#### 梯度增强回归树

梯度增强回归树是对任意可微损失函数进行增强的推广。梯度增强回归树是一种精确有效的现成程序，可用于回归和分类问题。梯度树增强模型被广泛应用于网络搜索排名和生态等领域。其优点是可以对混合类型数据的自然处理、有较好的预测能力以及处理输出空间异常值时的鲁棒性。

![](https://upload-images.jianshu.io/upload_images/3220531-4c3c45a7cfbf2864.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在这里，我们先定义好一个梯度增强树🌲回归器，然后用黄转头🧱可视化并评分，最后用散点图展示了419个有效数据中球员真实工资与预测薪水之间的差距。


```python
# Instantiate the model and visualizer
gbr = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                    learning_rate=0.1, loss='huber',
                                    random_state=1)
visualizer = PredictionError(gbr)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data

# Prediction plot
predictions = gbr.predict(X)
prd_gbr = predictions.astype(int)
plt.figure(figsize=(6,4))
x = range(len(prd_gbr))
plt.grid(ls=':', lw=1)
plt.scatter(x, prd_gbr, color='#FEB64D', marker='o', facecolor='None', lw=2)
plt.scatter(x, y, color='#9287E7', marker='o')
label = ["Prediction", "Actual"]
plt.legend(label, loc=2, markerscale=0.85, ncol = 1, fontsize=10, framealpha=1)

plt.title('Gradient Boosting Regression', fontsize=12)
plt.savefig("./fig/reg/gbr.png")
plt.show()

# Show score
print("Model Score: ",gbr.score(X_test, y_test))
```


![](https://upload-images.jianshu.io/upload_images/3220531-9ff6cb0e9670009b.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



![](https://upload-images.jianshu.io/upload_images/3220531-beabaefc744c028b.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


    Model Score:  0.5696922511542333

---

## 方法评估

用了这么一大堆回归模型和方法，究竟那个更适合我们这个问题呢？

![](https://upload-images.jianshu.io/upload_images/3220531-f3cc736584ce5de1.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们用`sklearn.metrics`中的各种指标评估了一下每种模型的表现。我们发现，大部分情况下，**非常随机回归树和梯度增强回归树🌲表现最好——在获得最高“R\_2 Score”的同时拥有“最小的误差”**。

### 度量标准

首先我们在这里确定好我们的度量标准。Luckily，`sklearn`为我们提供了很多常用的度量回归标准，我们在这里用了几个比较常用的metrics作demo。

![](https://upload-images.jianshu.io/upload_images/3220531-9caef4e081e552b2.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


```python
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
```

### 评估

#### R2 得分

`r2_score`函数计算决定系数，通常表示为$R^2$。它表示模型中自变量解释的方差的比例，提供了拟合优度的指标，可以衡量模型预测不可见样本的可能性。因此方差数据集相关，$R^2$可能不是有意义的比较在不同的数据集。最好的分数是1.0，也可以是负数。如果一个常数模型总是预测的期望值，无视输入功能，会得到一个$R^2=0$。`r2_score`具体计算公式是：

$$R^2(y,\hat{y})=1-\frac{\sum_{i=1}^n (y_i-\hat{y_i})^2}{\sum_{i=1}^n (y_i-\overline{y})^2}.$$

我们先将所有的模型进行评估，放到一个list `metric_r2_score`中，然后画出柱形图来观察每种模型的`r2_score`。



```python
metric_r2_score = []
metric_r2_score.append(r2_score(y, prd_lr))
metric_r2_score.append(r2_score(y, prd_lasso))
metric_r2_score.append(r2_score(y, prd_ridge))
metric_r2_score.append(r2_score(y, prd_ridgeCV))
metric_r2_score.append(r2_score(y, prd_svr))
metric_r2_score.append(r2_score(y, prd_decision_tree))
metric_r2_score.append(r2_score(y, prd_knn))
metric_r2_score.append(r2_score(y, prd_rfr))
metric_r2_score.append(r2_score(y, prd_etr))
metric_r2_score.append(r2_score(y, prd_gbr))
metric_r2_score.append(r2_score(y, prd_enet))
```


```python
x_ax = range(len(metric_r2_score))
plt.figure(figsize=(6,4))
plt.grid(ls=':', lw=2)
plt.title('Evaluation')
plt.xlabel('Regression')
plt.ylabel('R2_Score')
plt.xticks(x_ax, ('linear','lasso','ridge','ridgeCV','svr','dec-tree','knn','rfr','etr','gbr','enet'))
plt.bar(x_ax, metric_r2_score, hatch='o', \
        color=['#409cfc','#32d3eb','#5bc49f','#feb64d','#ff7c7c','#9287e7'], \
        edgecolor=None, alpha=.7, width=.6)
plt.savefig("./fig/evl/R2_scores.png")
plt.show()
```


![](https://upload-images.jianshu.io/upload_images/3220531-30f54cf0b7c30b05.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


经过观察我们看到，非常随机回归树有最高的`r2_score`，说明模型有最小的方差、预估结果最稳定。

#### 平均方差

`mean_squared_error`函数计算均方误差，即与平方误差或损失的期望值相对应的风险度量。计算公式为：

$$MSE(y,\hat{y})=\frac{1}{n_{samples}}\sum_{i=0}^{n_{samples}-1}(y_i-\hat{y_i})^2.$$

我们先将所有的模型进行评估，放到一个list `errors`中，然后画出柱形图来观察每种模型的`mean_squared_error`。


```python
errors = []
errors.append(mean_squared_error(y, prd_lr))
errors.append(mean_squared_error(y, prd_lasso))
errors.append(mean_squared_error(y, prd_ridge))
errors.append(mean_squared_error(y, prd_ridgeCV))
errors.append(mean_squared_error(y, prd_svr))
errors.append(mean_squared_error(y, prd_decision_tree))
errors.append(mean_squared_error(y, prd_knn))
errors.append(mean_squared_error(y, prd_rfr))
errors.append(mean_squared_error(y, prd_etr))
errors.append(mean_squared_error(y, prd_gbr))
errors.append(mean_squared_error(y, prd_enet))
```


```python
x_ax = range(len(errors))
plt.figure(figsize=(6,4))
plt.grid(ls=':', lw=2)
plt.title('Evaluation')
plt.xlabel('Regression')
plt.ylabel('Mean Square Error')
plt.xticks(x_ax, ('linear','lasso','ridge','ridgeCV','svr','dec-tree','knn','rfr','etr','gbr','enet'))
plt.bar(x_ax, errors, hatch='*', \
        color=['#409cfc','#32d3eb','#5bc49f','#feb64d','#ff7c7c','#9287e7'], \
        edgecolor=None, alpha=.7, width=.6)
plt.savefig("./fig/evl/mean_squares.png")
plt.show()
```


![](https://upload-images.jianshu.io/upload_images/3220531-d68a587e4865a6d1.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


经过观察我们看到，非常随机回归树有最小的`mean_squared_error`，说明模型有最小的方差、预估结果最稳定。

#### 平均绝对误差

`mean_absolute_error`函数计算平均绝对误差，这是一个风险度量，对应于绝对误差损失或`l1-norm`损失的期望值。计算公式为：

$$MAE(y,\hat{y})=\frac{1}{n_{samples}}\sum_{i=0}^{n_{samples}-1}|y_i-\hat{y_i}|.$$

我们先将所有的模型进行评估，放到一个list `ab_errors`中，然后画出柱形图来观察每种模型的`mean_absolute_error`。


```python
ab_errors = []
ab_errors.append(mean_absolute_error(y, prd_lr))
ab_errors.append(mean_absolute_error(y, prd_lasso))
ab_errors.append(mean_absolute_error(y, prd_ridge))
ab_errors.append(mean_absolute_error(y, prd_ridgeCV))
ab_errors.append(mean_absolute_error(y, prd_svr))
ab_errors.append(mean_absolute_error(y, prd_decision_tree))
ab_errors.append(mean_absolute_error(y, prd_knn))
ab_errors.append(mean_absolute_error(y, prd_rfr))
ab_errors.append(mean_absolute_error(y, prd_etr))
ab_errors.append(mean_absolute_error(y, prd_gbr))
ab_errors.append(mean_absolute_error(y, prd_enet))
```


```python
x_ax = range(len(ab_errors))
plt.figure(figsize=(6,4))
plt.grid(ls=':', lw=2)
plt.title('Evaluation')
plt.xlabel('Regression')
plt.ylabel('Mean Absolute Error')
plt.xticks(x_ax, ('linear','lasso','ridge','ridgeCV','svr','dec-tree','knn','rfr','etr','gbr','enet'))
plt.bar(x_ax, ab_errors, hatch='x', \
        color=['#409cfc','#32d3eb','#5bc49f','#feb64d','#ff7c7c','#9287e7'], \
        edgecolor=None, alpha=.7, width=.6)
plt.savefig("./fig/evl/absolute_squares.png")
plt.show()
```


![](https://upload-images.jianshu.io/upload_images/3220531-26944dc72553d063.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


经过观察我们看到，非常随机回归树有最小的`mean_absolute_error`，说明模型有最小的绝对误差、预估结果最稳定。

#### 解释方差的分数

解释变量度量数学模型对给定数据集的变化所占的比例，计算公式为：

$$EXV(y,\hat{y})=1-\frac{Var\{y-\hat{y}\}}{Var\{y\}}.$$

我们先将所有的模型进行评估，放到一个list `explained_vars`中，然后画出柱形图来观察每种模型的`explained_variance_score`。


```python
explained_vars = []
explained_vars.append(explained_variance_score(y, prd_lr))
explained_vars.append(explained_variance_score(y, prd_lasso))
explained_vars.append(explained_variance_score(y, prd_ridge))
explained_vars.append(explained_variance_score(y, prd_ridgeCV))
explained_vars.append(explained_variance_score(y, prd_svr))
explained_vars.append(explained_variance_score(y, prd_decision_tree))
explained_vars.append(explained_variance_score(y, prd_knn))
explained_vars.append(explained_variance_score(y, prd_rfr))
explained_vars.append(explained_variance_score(y, prd_etr))
explained_vars.append(explained_variance_score(y, prd_gbr))
explained_vars.append(explained_variance_score(y, prd_enet))
```


```python
x_ax = range(len(explained_vars))
plt.figure(figsize=(6,4))
plt.grid(ls=':', lw=2)
plt.title('Evaluation')
plt.xlabel('Regression')
plt.ylabel('Explained Variance Score')
plt.xticks(x_ax, ('linear','lasso','ridge','ridgeCV','svr','dec-tree','knn','rfr','etr','gbr','enet'))
plt.bar(x_ax, explained_vars, hatch='.', \
        color=['#409cfc','#32d3eb','#5bc49f','#feb64d','#ff7c7c','#9287e7'], \
        edgecolor=None, alpha=.7, width=.6)
plt.savefig("./fig/evl/ev_scores.png")
plt.show()
```


![](https://upload-images.jianshu.io/upload_images/3220531-1811763d8281d7d6.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


经过观察我们看到，非常随机回归树有最小的`explained_variance_score`，说明模型有最小的绝对误差、预估结果最稳定。

---

## 结论与展望

我们在这个实验中用微软的**Azure云☁️机器学习平台**工具对NBA球员的能力进行评估，并对他们的薪水进行预测。我们采用了`sklearn`轮子中的多种回归模型，预测出的结果也有一定可信性。

![](https://upload-images.jianshu.io/upload_images/3220531-f8e8126d4527c42c.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**但是！但是！但是！**

我们还是不能将这些回归用于工业用途，因为与真实世界相比，这些模型还是有失公允。误差的来源可能还会来自以下方面。

有时候，数据并不匹配能力，很多球员的价值无法仅仅用数据来衡量（*当然了，薪资和能力有时候也不成正相关*）。有些球星常规赛划水，为了保持充沛体能不停轮休，到了季后赛，倒是掀开棺材板；有些常规赛巨星到了季后赛就萎靡不振，常年一轮二轮游🎣。

![](https://upload-images.jianshu.io/upload_images/3220531-b2da197a4114d37e.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

数据的成色也不尽相同，有些球队拥有多个球星，他们的发挥可能受到球权限制。对于这些球员，即使数据缩水，他们在自由市场中依然还是香饽饽，很有可能拿到大合同。

![](https://upload-images.jianshu.io/upload_images/3220531-1974ac8076521a49.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

除此之外，NBA联盟中的贫富差距还是蛮大的，很多超级巨星富可敌国，有些角色球员还在摸打滚爬。我们的预测模型在预测极端情况时，比较保守。也不难理解，有时候，得到一个顶级巨星的价值是无与伦比的，这种球星的价值对球队现在甚至历史而言都是不可估量的。

![](https://upload-images.jianshu.io/upload_images/3220531-368595c93992958d.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

另外，随着工资帽水涨船高，球员都“共同富裕了”，甚至在同一个赛季都能看见明显涨幅。因此我们的模型的预测值大部分没有达到球员真实薪水。除此之外，好的薪水离不开经纪人的努力。对于一些数据并不显眼的球员，很多大牌经纪人往往能带来更多红利。

![](https://upload-images.jianshu.io/upload_images/3220531-9e868bb33afa2162.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

球员工资不仅仅取决于场上表现，还跟球员影响力有关。影响力巨大的球星，就算是数据表现不好，还是能拿得到大合同，因此仅仅横向考虑数据因素是远远不够的。数据集存在的天然缺陷决定了我们的方法也不能普适于各种情况。

![](https://upload-images.jianshu.io/upload_images/3220531-3d637717f78eee13.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

另外，NBA还出台了很多rules，比如伯德条款、罗斯条款等等，工资还会受到各种条约限制。NBA的薪资规则是十分复杂的，因此，就算能力再强、数据再好，也得遵循特定的工资规则。仅仅靠回归模型进行预测定然是与真实情况相去甚远的了。

![](https://upload-images.jianshu.io/upload_images/3220531-cc1cc7c20d5fccbc.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---