---
layout: default
title: Publications
permalink: /publications/
---

<div class="container" style="margin-top: 50px; max-width: 1000px;">
  <h2 style="margin-bottom: 40px; color: #333; font-weight: 300;">Publications</h2>
  
  <ul class="publications-list" style="list-style: none; padding: 0;">
    {% assign pubs = site.publications %}
    {% if pubs.size > 0 %}
      {% assign sorted_pubs = pubs | sort: 'date' | reverse %}
      {% for pub in sorted_pubs %}
      <li style="margin-bottom: 50px;">
        <h3 style="margin-bottom: 12px; font-weight: 500; line-height: 1.3;">
          <a href="{{ pub.paper }}" target="_blank" style="color: #007bff; text-decoration: none;">{{ pub.title }}</a>
        </h3>
        
        <p style="font-size: 1.4rem; color: #888; margin-bottom: 12px;">{{ pub.date | date: "%B %Y" }}</p>
        
        <div class="pub-authors" style="font-size: 1.5rem; color: #444; margin-bottom: 15px; line-height: 1.6;">
          {% assign authors = pub.authors | split: ", " %}
          {% for author in authors %}
            {% if author contains "Azime" %}
              <strong style="text-decoration: underline; color: #222;">{{ author }}</strong>{% unless forloop.last %}, {% endunless %}
            {% else %}
              {{ author }}{% unless forloop.last %}, {% endunless %}
            {% endif %}
          {% endfor %}
        </div>

        <div class="pub-summary" style="font-size: 1.5rem; color: #555; line-height: 1.7; margin-bottom: 25px; text-align: justify;">
          {{ pub.content | strip_html | truncate: 600 }}
        </div>
        
        <div class="pub-links" style="display: flex; gap: 15px;">
          <a href="{{ pub.paper }}" target="_blank" class="btn-arxiv">
            <i class="bi bi-file-earmark-text" style="font-size: 1.6rem; margin-bottom: 4px;"></i>
            <span style="font-size: 1.1rem; display: block; font-weight: 500;">Abstract</span>
          </a>
          <a href="{{ pub.paper | replace: 'abs', 'pdf' }}" target="_blank" class="btn-arxiv">
            <i class="bi bi-file-pdf" style="font-size: 1.6rem; margin-bottom: 4px;"></i>
            <span style="font-size: 1.1rem; display: block; font-weight: 500;">View PDF</span>
          </a>
        </div>
      </li>
      {% endfor %}
    {% else %}
      <li style="text-align: center; padding: 100px 0;">
        <p style="font-style: italic; color: #888;">No publications found since 2019.</p>
      </li>
    {% endif %}
  </ul>
</div>

<style>
  .btn-arxiv {
    background: #333;
    color: #fff !important;
    padding: 10px 20px;
    border-radius: 6px;
    text-align: center;
    min-width: 100px;
    text-decoration: none !important;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }
  .btn-arxiv:hover {
    background: #007bff;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
  }
  .dark .btn-arxiv {
    background: #444;
  }
  .dark .btn-arxiv:hover {
    background: #007bff;
  }
  .dark .pub-authors { color: #ccc; }
  .dark .pub-authors strong { color: #fff; }
  .dark .pub-summary { color: #bbb; }
  .dark h2 { color: #eee; }
</style>
