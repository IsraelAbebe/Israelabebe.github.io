---
layout: home
title: Israel A. Azime
<!-- subtitle: Senior Machine Learning Engineer | Software Programmer -->
tags: uni-saarland machine-learning 
---




<hr>
<br>





I am currently a PhD student and research assistant at [Saarland University](https://www.uni-saarland.de/en/home.html), with over four years of experience in the machine learning industry. My journey in the field includes roles such as Applied Research Data Scientist at [Iquartic](https://iquartic.com/) and Applied ML Developer at [SingularityNet/iCog-Labs](https://singularitynet.io/).

Academically, I hold an MSc in Mathematical Sciences – Machine Intelligence from [AIMS-AMMI](https://aimsammi.org/) and a Bachelor of Science in Software Engineering from [Addis Ababa Institute of Technology](http://www.aait.edu.et/). I am an active member of the [Masakhane](https://www.masakhane.io/) and [ETHIO NLP](https://ethionlp.github.io/) community.

My research interest includes natural language processing, high-performance computing in AI, multimodal learning and the application of deep learning.



<h2 align='center'>News</h2>



{% for page in site.pages reversed %}
{% if page.url contains '/pages/news/' and page.title %}
<p>{{ page.date | date: "%B %d, %Y" }} 🗞️ {{page.role}} <i> <a href="{{page.link}}">{{ page.title }}</a></i></p> 
{% endif %}
{% endfor %}

