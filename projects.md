---
layout: default
title: Projects
permalink: /projects/
---

<div class="container" style="margin-top: 50px;">
  <h2 style="margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 10px;">GitHub Projects</h2>
  
  <div style="text-align: center; margin-bottom: 40px;">
    <img src="http://ghchart.rshah.org/IsraelAbebe" alt="IsraelAbebe's Github chart" style="max-width: 100%;" />
  </div>

  <div id="github-projects" class="row">
    <p style="text-align: center; font-style: italic; color: #888;">Fetching repositories from GitHub...</p>
  </div>
</div>

<script>
  (function() {
    const username = 'IsraelAbebe';
    const apiUrl = `https://api.github.com/users/${username}/repos?sort=updated&per_page=100`;

    fetch(apiUrl)
      .then(response => response.json())
      .then(repos => {
        const sortedRepos = repos
          .filter(repo => !repo.fork && repo.description)
          .sort((a, b) => b.stargazers_count - a.stargazers_count);

        let html = '';
        sortedRepos.forEach(repo => {
          html += `
            <div class="col-xs-12 col-sm-6" style="margin-bottom: 30px;">
              <div class="project-card" style="border: 1px solid #eee; padding: 25px; border-radius: 12px; height: 100%; transition: all 0.3s ease; overflow: hidden; display: flex; flex-direction: column;">
                <h4 style="margin-top: 0; margin-bottom: 12px;">
                  <a href="${repo.html_url}" target="_blank" style="font-weight: 600; color: #007bff; text-decoration: none;">${repo.name}</a>
                </h4>
                <p style="font-size: 1.4rem; color: #666; margin-bottom: 20px; line-height: 1.6; flex-grow: 1;">${repo.description || 'No description available.'}</p>
                
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: auto;">
                  <div style="font-size: 1.2rem; color: #888; display: flex; align-items: center; gap: 15px;">
                    <span><i class="fas fa-circle" style="color: #3182ce; font-size: 0.8rem; margin-right: 5px;"></i> ${repo.language || 'Mixed'}</span>
                    <span><i class="far fa-star"></i> ${repo.stargazers_count}</span>
                  </div>
                  <a href="${repo.html_url}" target="_blank" class="btn-github-link">
                    <i class="fab fa-github"></i> View Repo
                  </a>
                </div>
              </div>
            </div>
          `;
        });

        document.getElementById('github-projects').innerHTML = html || '<p style="text-align: center;">No projects found.</p>';
      })
      .catch(err => {
        document.getElementById('github-projects').innerHTML = '<p style="text-align: center;">Error loading projects from GitHub.</p>';
        console.error(err);
      });
  })();
</script>

<style>
  .project-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 24px rgba(0,0,0,0.1);
    border-color: #007bff !important;
  }
  .btn-github-link {
    background: #f6f8fa;
    border: 1px solid #d1d5da;
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 1.2rem;
    color: #24292e !important;
    text-decoration: none !important;
    font-weight: 500;
    transition: all 0.2s;
  }
  .btn-github-link:hover {
    background: #007bff;
    color: #fff !important;
    border-color: #007bff;
  }
  .dark .project-card {
    border-color: #333 !important;
    background: #1a1a1a;
  }
  .dark .project-card:hover {
    box-shadow: 0 12px 24px rgba(255,255,255,0.05);
  }
  .dark .project-card p { color: #aaa; }
  .dark .project-card h4 a { color: #66b2ff; }
  .dark .btn-github-link {
    background: #2d333b;
    border-color: #444c56;
    color: #adbac7 !important;
  }
</style>
