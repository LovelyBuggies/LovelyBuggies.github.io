---
layout:     post
title:      5 Mins build Vue.js on your Mac!
subtitle:   
date:       2019-04-15
author:     Nino Lau
header-img: img/post-bg-os-metro.jpg
catalog: true
tags:
    - 知识介绍
---

As the project manager 👨🏻‍💻 of our [Make $pare Money 💰]([https://github.com/make-money-sysu/Make-Money](https://github.com/make-money-sysu/Make-Money)
) project, I began to touch the front-end tasks for the first time today. Informed that we will use [Vue.js]([https://www.baidu.com/link?url=Xnh9a8MWdATXnGn2MSt8w0bf5weix8ZMr_25X-J1CrC&ck=8364.1.62.147.181.160.186.560&shh=www.baidu.com&sht=64075107_1_dg&wd=&eqid=c1c9986b000204fe000000065cb44ee4](https://www.baidu.com/link?url=Xnh9a8MWdATXnGn2MSt8w0bf5weix8ZMr_25X-J1CrC&ck=8364.1.62.147.181.160.186.560&shh=www.baidu.com&sht=64075107_1_dg&wd=&eqid=c1c9986b000204fe000000065cb44ee4)
) in our front-end construction 🛠. In order to keep track of their process, I need to learn Vue.js at first! 😛

## Introduce of Vue

> Vue is a progressive framework for building user interfaces. It is designed from the ground up to be incrementally adoptable, and can easily scale between a library and a framework depending on different use cases. It consists of an approachable core library that focuses on the view layer only, and an ecosystem of supporting libraries that helps you tackle complexity in large Single-Page Applications.

As one of the most popular front-end architecture, Vue.js enjoys many merits ([merits here]([https://github.com/vuejs/vue](https://github.com/vuejs/vue)
)). 💗 You can also view some Vue examples [here]([https://vuejs.org/v2/examples/index.html](https://vuejs.org/v2/examples/index.html)
). 😊

## Vue.js Set-up

It's pretty simple to build a Vue.js environment on your Mac if you are familiar with stuff like `node`, `homebrew` ... Let's start now!  🎉

Upgrade your brew and node.

```shell
$ brew upgrade
$ brew upgrade node
```
Install Vue client.

``` shell
$ sudo npm install -g vue-cli
$ vue -V
```

If you have version shown on your screen, congratulations, you have successfully install Vue!  🎉

Let's create our first Vue project!

``` shell
$ vue init webpack firstvueproject   # you can change 'firstvueproject' to your project name!
```

As for the other dependent modules of the Vue project, we installed them separately according to the project requirements. *You can set them like mine.*

![](https://upload-images.jianshu.io/upload_images/3220531-5211872cd13334fa.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Well done! 👍Now we can see a project in our floder.

![](https://upload-images.jianshu.io/upload_images/3220531-ba4dc198e2a69717.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Now we will not take the structure into consideration. Let's run our project now! 

``` shell 
$ npm run dev
```
![](https://upload-images.jianshu.io/upload_images/3220531-f07d612d98126fb6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Open http://localhost:8080 on your browser, we have made our first Vue project!

![](https://upload-images.jianshu.io/upload_images/3220531-c30e05e9ebfb6ecb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Amazing!!!  🤣

I will going to tell some further tricks of Vue in the future!  😜