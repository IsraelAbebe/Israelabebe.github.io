function applyTheme() {
    const theme = localStorage.getItem('theme');
    if (theme === 'dark') {
        document.body.classList.add('dark');
        document.documentElement.classList.add('dark');
    } else {
        document.body.classList.remove('dark');
        document.documentElement.classList.remove('dark');
    }
}

applyTheme();

function toggleTheme() {
    if (document.documentElement.classList.contains('dark')) {
        localStorage.setItem('theme', 'light');
    } else {
        localStorage.setItem('theme', 'dark');
    }
    applyTheme();
}

function downloadCV(url) {
    const modal = document.getElementById('cv-modal');
    const loading = document.getElementById('cv-loading');
    const success = document.getElementById('cv-success');
    
    modal.style.display = 'flex';
    loading.style.display = 'block';
    success.style.display = 'none';

    setTimeout(() => {
        loading.style.display = 'none';
        success.style.display = 'block';
        
        const link = document.createElement('a');
        link.href = url;
        link.download = 'Israel_Azime_CV.pdf';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }, 1500);
}

function closeCVModal() {
    document.getElementById('cv-modal').style.display = 'none';
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('cv-modal');
    if (event.target == modal) {
        modal.style.display = "none";
    }
}
