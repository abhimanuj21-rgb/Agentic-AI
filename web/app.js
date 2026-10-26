/**
 * Agentic RAG Coder - Frontend Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  // --- Tab Navigation ---
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabPanels = document.querySelectorAll('.tab-panel');

  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab');
      tabButtons.forEach(b => b.classList.remove('active'));
      tabPanels.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const panel = document.getElementById(targetId);
      if (panel) panel.classList.add('active');

      if (targetId === 'knowledge-view') loadFiles();
      if (targetId === 'agents-view') loadHistory();
    });
  });

  // --- Initial System Data Load ---
  fetchSystemStats();
  loadFiles();

  // --- Chat Setup ---
  const chatForm = document.getElementById('chat-form');
  const userInput = document.getElementById('user-input');
  const chatMessages = document.getElementById('chat-messages');
  const stepper = document.getElementById('agent-stepper');
  const memoryCheckbox = document.getElementById('memory-checkbox');

  // Suggested Prompts Chips
  document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const prompt = chip.getAttribute('data-prompt');
      userInput.value = prompt;
      userInput.focus();
    });
  });

  // Chat Form Submit
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = userInput.value.trim();
    if (!query) return;

    // Append User Message Card
    appendMessageCard('user', 'You', query);
    userInput.value = '';

    // Show Stepper Animation
    stepper.style.display = 'block';
    resetStepper();
    simulateStepperFlow();

    try {
      const resp = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: query,
          use_memory: memoryCheckbox.checked
        })
      });

      const data = await resp.json();
      stepper.style.display = 'none';

      if (resp.ok) {
        appendMessageCard('assistant', 'Crew Lead & Senior Coder', data.response, data.citations);
        fetchSystemStats();
      } else {
        appendMessageCard('assistant', 'System Notice', `Error: ${data.detail || 'Execution failed.'}`);
      }
    } catch (err) {
      stepper.style.display = 'none';
      appendMessageCard('assistant', 'System Error', `Network or server error: ${err.message}`);
    }
  });

  // --- Knowledge Ingestion & Sync ---
  const btnSyncIndex = document.getElementById('btn-sync-index');
  const btnResetIndex = document.getElementById('btn-reset-index');
  const addDocForm = document.getElementById('add-doc-form');

  btnSyncIndex.addEventListener('click', async () => {
    btnSyncIndex.disabled = true;
    btnSyncIndex.innerHTML = '<span>Indexing...</span>';
    try {
      const res = await fetch('/api/index', { method: 'POST' });
      const data = await res.json();
      alert(`Indexing complete! ${data.chunks_indexed} new chunks added.`);
      loadFiles();
      fetchSystemStats();
    } catch (err) {
      alert(`Sync failed: ${err.message}`);
    } finally {
      btnSyncIndex.disabled = false;
      btnSyncIndex.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/>
        </svg>
        <span>Sync / Re-Index Now</span>`;
    }
  });

  btnResetIndex.addEventListener('click', async () => {
    if (!confirm('Are you sure you want to wipe the vector index and re-index from scratch?')) return;
    try {
      const res = await fetch('/api/index', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reset: true })
      });
      const data = await res.json();
      alert('Index reset and rebuilt successfully!');
      loadFiles();
      fetchSystemStats();
    } catch (err) {
      alert(`Reset failed: ${err.message}`);
    }
  });

  addDocForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const filename = document.getElementById('new-doc-name').value.trim();
    const content = document.getElementById('new-doc-content').value;

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename, content })
      });
      const data = await res.json();
      alert(`Document '${filename}' saved and indexed!`);
      addDocForm.reset();
      loadFiles();
      fetchSystemStats();
    } catch (err) {
      alert(`Error saving document: ${err.message}`);
    }
  });

  // --- Vector Store Query Explorer ---
  const vectorSearchForm = document.getElementById('vector-search-form');
  const vectorQueryInput = document.getElementById('vector-query-input');
  const topKSlider = document.getElementById('search-top-k');
  const threshSlider = document.getElementById('search-thresh');
  const topKLabel = document.getElementById('top-k-label');
  const threshLabel = document.getElementById('thresh-label');
  const resultsContainer = document.getElementById('vector-results-container');

  topKSlider.addEventListener('input', () => { topKLabel.textContent = topKSlider.value; });
  threshSlider.addEventListener('input', () => { threshLabel.textContent = threshSlider.value; });

  vectorSearchForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = vectorQueryInput.value.trim();
    if (!query) return;

    resultsContainer.innerHTML = '<div class="empty-state"><p>Searching vector embeddings...</p></div>';

    try {
      const res = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          top_k: parseInt(topKSlider.value, 10),
          similarity_threshold: parseFloat(threshSlider.value)
        })
      });

      const data = await res.json();
      renderVectorResults(data.results || []);
    } catch (err) {
      resultsContainer.innerHTML = `<div class="empty-state"><p style="color:var(--accent-rose)">Search error: ${err.message}</p></div>`;
    }
  });

  // --- Session Memory Management ---
  const btnClearSession = document.getElementById('btn-clear-session');
  btnClearSession.addEventListener('click', async () => {
    if (!confirm('Clear all conversation turns in session memory?')) return;
    try {
      await fetch('/api/clear-memory', { method: 'POST' });
      alert('Session memory cleared.');
      loadHistory();
    } catch (err) {
      alert(`Error: ${err.message}`);
    }
  });

  // --- Helper Functions ---

  async function fetchSystemStats() {
    try {
      const res = await fetch('/api/stats');
      if (!res.ok) return;
      const stats = await res.json();

      document.getElementById('chunks-count-text').textContent = `${stats.total_chunks} Chunks`;
      document.getElementById('stat-chunks-count').textContent = stats.total_chunks;
      document.getElementById('stat-persist-dir').textContent = stats.persist_directory || './data/chroma_db';
      document.getElementById('stat-engine-name').textContent = stats.backend || 'ChromaDB';
      document.getElementById('provider-text').textContent = (stats.default_provider || 'Groq').toUpperCase();
    } catch (err) {
      console.warn('Stats fetch warning:', err);
    }
  }

  async function loadFiles() {
    try {
      const res = await fetch('/api/files');
      if (!res.ok) return;
      const data = await res.json();
      const files = data.files || [];

      document.getElementById('stat-files-count').textContent = files.length;
      const tbody = document.getElementById('files-table-body');

      if (!files.length) {
        tbody.innerHTML = '<tr><td colspan="5" class="empty-state">No local files found in ./data/raw.</td></tr>';
        return;
      }

      tbody.innerHTML = files.map(f => `
        <tr>
          <td><strong>${escapeHtml(f.filename)}</strong><br><span style="font-size:0.75rem;color:var(--text-muted)">${escapeHtml(f.file_path)}</span></td>
          <td><span class="citation-badge">${escapeHtml(f.extension)}</span></td>
          <td>${formatBytes(f.size)}</td>
          <td>${new Date(f.last_modified).toLocaleString()}</td>
          <td><span class="hash-badge">SHA-256 ✓</span></td>
        </tr>
      `).join('');
    } catch (err) {
      console.error('Error loading files:', err);
    }
  }

  async function loadHistory() {
    try {
      const res = await fetch('/api/history');
      if (!res.ok) return;
      const data = await res.json();
      const history = data.history || [];
      const box = document.getElementById('session-history-container');

      if (!history.length) {
        box.innerHTML = '<div class="empty-state"><p>No prior turns in this session.</p></div>';
        return;
      }

      box.innerHTML = history.map((t, idx) => `
        <div class="history-turn-card">
          <div class="turn-header">
            <span>Turn #${idx + 1}</span>
            <span>${new Date(t.timestamp).toLocaleTimeString()}</span>
          </div>
          <div class="turn-q">Q: ${escapeHtml(t.query)}</div>
          <div class="turn-a">${escapeHtml(t.response.substring(0, 240))}...</div>
        </div>
      `).join('');
    } catch (err) {
      console.error('Error loading history:', err);
    }
  }

  function appendMessageCard(role, senderName, content, citations) {
    const card = document.createElement('div');
    card.className = `message-card ${role}`;

    const avatar = role === 'user' ? '👤' : '🤖';
    const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    let citationsHtml = '';
    if (citations && citations.length) {
      citationsHtml = `
        <div class="citations-box">
          <h5>Grounded Sources & Citations</h5>
          <ul class="citations-list">
            ${citations.map(c => `
              <li class="citation-item">
                <span class="citation-badge">${escapeHtml(c.type || 'FILE')}</span>
                <span>${escapeHtml(c.source)}</span>
                ${c.similarity ? `<span style="color:var(--accent-emerald);font-size:0.75rem">(${Math.round(c.similarity * 100)}% match)</span>` : ''}
              </li>
            `).join('')}
          </ul>
        </div>
      `;
    }

    card.innerHTML = `
      <div class="card-avatar">${avatar}</div>
      <div class="card-body">
        <div class="card-header">
          <span class="agent-name">${escapeHtml(senderName)}</span>
          <span class="timestamp">${now}</span>
        </div>
        <div class="markdown-content">
          ${renderMarkdown(content)}
          ${citationsHtml}
        </div>
      </div>
    `;

    chatMessages.appendChild(card);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function renderVectorResults(results) {
    if (!results.length) {
      resultsContainer.innerHTML = '<div class="empty-state"><p>No chunks matched your query with the current threshold.</p></div>';
      return;
    }

    resultsContainer.innerHTML = results.map((r, i) => {
      const matchPct = Math.round((r.similarity || 0) * 100);
      return `
        <div class="result-card">
          <div class="result-header">
            <span class="result-source">[#${i + 1}] ${escapeHtml(r.metadata?.source_file || 'unknown')} (${escapeHtml(r.chunk_id)})</span>
            <div class="result-score-meter">
              <span>${matchPct}% Match</span>
              <div class="meter-bar"><div class="meter-fill" style="width: ${matchPct}%;"></div></div>
            </div>
          </div>
          <div class="result-snippet">${escapeHtml(r.content || '')}</div>
        </div>
      `;
    }).join('');
  }

  function simulateStepperFlow() {
    const sManager = document.getElementById('step-manager');
    const sRetriever = document.getElementById('step-retriever');
    const sResearcher = document.getElementById('step-researcher');
    const sCoder = document.getElementById('step-coder');

    sManager.classList.add('active');
    setTimeout(() => {
      sManager.classList.remove('active');
      sManager.classList.add('completed');
      sRetriever.classList.add('active');
    }, 400);

    setTimeout(() => {
      sRetriever.classList.remove('active');
      sRetriever.classList.add('completed');
      sResearcher.classList.add('active');
    }, 900);

    setTimeout(() => {
      sResearcher.classList.remove('active');
      sResearcher.classList.add('completed');
      sCoder.classList.add('active');
    }, 1400);
  }

  function resetStepper() {
    ['step-manager', 'step-retriever', 'step-researcher', 'step-coder'].forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.classList.remove('active', 'completed');
      }
    });
  }

  function renderMarkdown(md) {
    if (!md) return '';
    let html = escapeHtml(md);

    // Code blocks
    html = html.replace(/```([a-zA-Z0-9_]*)\n([\s\S]*?)```/g, (match, lang, code) => {
      return `<pre><code class="language-${lang}">${code}</code></pre>`;
    });

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Headers
    html = html.replace(/^### (.*$)/gim, '<h4 style="color:var(--accent-cyan);margin:0.8rem 0 0.4rem 0;">$1</h4>');
    html = html.replace(/^## (.*$)/gim, '<h3 style="color:#fff;margin:1rem 0 0.5rem 0;">$1</h3>');
    html = html.replace(/^# (.*$)/gim, '<h2 style="color:#fff;margin:1.2rem 0 0.6rem 0;">$1</h2>');

    // Bold & italic
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // Lists
    html = html.replace(/^\- (.*$)/gim, '<li style="margin-left:1.2rem;">$1</li>');

    // Paragraph line breaks
    html = html.replace(/\n\n/g, '<br><br>');
    return html;
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function formatBytes(bytes) {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }
});
