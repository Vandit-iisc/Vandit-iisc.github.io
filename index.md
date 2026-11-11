---
layout: default
---
<p class="intro">{{ site.description }}</p>

<ul class="post-list">
{% for post in site.posts %}
  <li>
    <span class="meta">{{ post.date | date: "%b %-d, %Y" }}</span>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    {% if post.description %}<p>{{ post.description }}</p>{% endif %}
  </li>
{% endfor %}
</ul>
