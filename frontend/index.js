const API_BASE_URL = 'http://localhost:8000';
let allCandidates = [];
let selectedCandidateId = null;
let currentEventSource = null;

// Fetch all candidates on page load
async function loadCandidates() {
  const listElement = document.getElementById('candidatesList');
  listElement.innerHTML = '<div class="loading"><div class="spinner"></div>Loading candidates...</div>';

  try {
    const response = await fetch(`${API_BASE_URL}/candidates`);
    const data = await response.json();
    allCandidates = data.candidates;
    renderCandidatesList(allCandidates);
  } catch (error) {
    listElement.innerHTML = `<div class="error">Failed to load candidates: ${error.message}</div>`;
  }
}

// Render candidates list
function renderCandidatesList(candidates) {
  const listElement = document.getElementById('candidatesList');

  if (candidates.length === 0) {
    listElement.innerHTML = '<p style="color: #999; text-align: center;">No candidates found</p>';
    return;
  }
  listElement.innerHTML = candidates.map(candidate => `
                <div class="candidate-item" onclick="selectCandidate('${candidate.candidate_id}')" data-id="${candidate.candidate_id}">
                    <h3>${candidate.candidate_name}</h3>
                    <p>#${candidate.candidate_id}</p>
                    <p>file:${candidate.file_name}</p>
                </div>
            `).join('');
}

// Select and load candidate details with streaming
async function selectCandidate(candidateId) {
  // Close previous stream if exists
  if (currentEventSource) {
    currentEventSource.close();
    currentEventSource = null;
  }

  selectedCandidateId = candidateId;

  document.querySelectorAll('.candidate-item').forEach(item => {
    // Update active state
    item.classList.remove('active');
    if (item.dataset.id === candidateId) {
      item.classList.add('active');
    }
  });

  // Load details with streaming
  const detailsElement = document.getElementById('candidate-details');
  detailsElement.classList.remove('empty');
  detailsElement.innerHTML = '<div class="loading"><div class="spinner"></div>Loading candidate details...</div>';

  try {
    // Create EventSource for SSE
    const eventSource = new EventSource(`${API_BASE_URL}/candidates/${encodeURIComponent(candidateId)}`);
    currentEventSource = eventSource;

    let metadata = null;
    let contentBuffer = '';

    eventSource.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);

        if (message.type === 'metadata') {
          metadata = message.data;
          // Initialize the details view
          detailsElement.innerHTML = `
            <div class="summary-content">
                <h1>${metadata.candidate_name}</h1>
                <p><strong>ID:</strong> ${metadata.candidate_id}</p>
                <p><strong>File:</strong> ${metadata.file_name}</p>
                <hr style="margin: 20px 0; border: 1px solid #e0e0e0;">
                <div id="streamingContent"></div>
            </div>
          `;
        } else if (message.type === 'content') {
          contentBuffer += message.data;
          updateStreamingContent(contentBuffer);
        } else if (message.type === 'done') {
          eventSource.close();
          currentEventSource = null;
        }
      } catch (error) {
        console.error('Error parsing SSE message:', error);
      }
    };

    eventSource.onerror = (error) => {
      console.error('EventSource error:', error);
      eventSource.close();
      currentEventSource = null;

      if (!metadata) {
        detailsElement.innerHTML = `<div class="error">Failed to load candidate details. Please try again.</div>`;
      }
    };

  } catch (error) {
    detailsElement.innerHTML = `<div class="error">Failed to load candidate details: ${error.message}</div>`;
  }
}

function updateStreamingAnswer(answer, candidates) {
  const answerElement = document.getElementById('streamingAnswer');
  const resultsListElement = document.getElementById('searchResultsList');

  if (answerElement) {
    // Convert markdown-like formatting
    let formattedAnswer = answer
      .replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      // Format agent output - Thought and Action on new lines with colors
      .replace(/Thought:/g, '<br><span class="agent-thought">Thought:</span>')
      .replace(/Action:/g, '<br><span class="agent-action">Action:</span>')
      .replace(/Action Input:/g, '<br><span class="agent-action">Action Input:</span>')
      .replace(/\n\n/g, '</p><p>');

    answerElement.innerHTML = `<p>${formattedAnswer}</p>`;
  }

  // Update candidates list if available
  if (resultsListElement && candidates && candidates.length > 0) {
    resultsListElement.innerHTML = `
      <button class="clear-filter-btn" onclick="clearFilter()">Clear Filter & Show All</button>
      ${candidates.map((candidate, idx) => `
        <div class="search-result-item" onclick="selectCandidate('${candidate.candidate_id}')">
          <div class="result-header">
            <h4>${idx + 1}. ${candidate.candidate_name}</h4>
            <span class="score-badge">${(candidate.score * 100).toFixed(1)}%</span>
          </div>
          <p class="result-meta">ID: ${candidate.candidate_id} | File: ${candidate.file_name}</p>
          <p class="result-content">${candidate.content.substring(0, 200)}${candidate.content.length > 200 ? '...' : ''}</p>
        </div>
      `).join('')}
    `;
  }
}

function updateStreamingContent(content) {
  const contentElement = document.getElementById('streamingContent');
  if (!contentElement) return;

  // Convert markdown-like formatting to HTML
  let formattedContent = content
    .replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    // Format agent output - Thought and Action on new lines with colors
    .replace(/Thought:/g, '<br><span class="agent-thought">Thought:</span>')
    .replace(/Action:/g, '<br><span class="agent-action">Action:</span>')
    .replace(/Action Input:/g, '<br><span class="agent-action">Action Input:</span>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/^- (.+)$/gm, '<li>$1</li>');

  // Wrap consecutive list items in ul tags
  formattedContent = formattedContent.replace(/(<li>.*?<\/li>\s*)+/g, (match) => {
    return '<ul>' + match + '</ul>';
  });

  contentElement.innerHTML = `<p>${formattedContent}</p>`;

  // Auto-scroll to bottom as content streams in
  contentElement.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

// Search functionality
document.getElementById('searchBox').addEventListener('input', (e) => {
  const searchTerm = e.target.value.toLowerCase();
  const filtered = allCandidates.filter(candidate =>
    candidate.candidate_name.toLowerCase().includes(searchTerm)
  );
  renderCandidatesList(filtered);
});

// Clean up on page unload
window.addEventListener('beforeunload', () => {
  if (currentEventSource) {
    currentEventSource.close();
  }
});

// Initialize
loadCandidates();

// Panel switching
function switchPanel(panelName) {
  document.querySelectorAll('.toggle-btn').forEach(btn => {
    btn.classList.remove('active');
    if (btn.dataset.panel === panelName) {
      btn.classList.add('active');
    }
  });

  document.querySelectorAll('.panel-content').forEach(panel => {
    panel.classList.remove('active');
  });

  if (panelName === 'details') {
    document.getElementById('candidate-details').classList.add('active');
  } else if (panelName === 'search') {
    document.getElementById('searchPanel').classList.add('active');
  }
}

// AI Search functionality
async function performAISearch() {
  const query = document.getElementById('aiSearchInput').value.trim();
  if (!query) return;

  const resultsElement = document.getElementById('aiSearchResults');
  const searchBtn = document.getElementById('aiSearchBtn');

  searchBtn.disabled = true;
  searchBtn.textContent = 'Searching...';
  resultsElement.innerHTML = '<div class="loading"><div class="spinner"></div>Searching candidates...</div>';

  try {
    const response = await fetch(`${API_BASE_URL}/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query, top_k: 10 })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let candidates = [];
    let answerBuffer = '';

    // Initialize results view
    resultsElement.innerHTML = `
      <div class="search-summary">
        <h3>Search Results</h3>
        <div id="streamingAnswer"></div>
      </div>
      <div class="search-results-list" id="searchResultsList"></div>
    `;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const message = JSON.parse(line.slice(6));

            if (message.type === 'metadata') {
              candidates = message.data.candidates;
            } else if (message.type === 'content') {
              answerBuffer += message.data;
              updateStreamingAnswer(answerBuffer, candidates);
            }
          } catch (e) {
            console.error('Error parsing SSE message:', e);
          }
        }
      }
    }

    searchBtn.disabled = false;
    searchBtn.textContent = 'Search';

  } catch (error) {
    console.error('Search error:', error);
    resultsElement.innerHTML = `<div class="error">Search failed: ${error.message}</div>`;
    searchBtn.disabled = false;
    searchBtn.textContent = 'Search';
  }
}

function displaySearchResults(data) {
  const resultsElement = document.getElementById('aiSearchResults');

  if (data.matching_candidates.length === 0) {
    resultsElement.innerHTML = '<p class="no-results">No candidates found matching your query.</p>';
    return;
  }

  const resultsHTML = `
    <div class="search-summary">
      <h3>Search Results</h3>
      <p>${data.answer}</p>
      <button class="clear-filter-btn" onclick="clearFilter()">Clear Filter & Show All</button>
    </div>
    <div class="search-results-list">
      ${data.matching_candidates.map((candidate, idx) => `
        <div class="search-result-item" onclick="selectCandidate('${candidate.candidate_id}')">
          <div class="result-header">
            <h4>${idx + 1}. ${candidate.candidate_name}</h4>
            <span class="score-badge">${(candidate.score * 100).toFixed(1)}%</span>
          </div>
          <p class="result-meta">ID: ${candidate.candidate_id} | File: ${candidate.file_name}</p>
          <p class="result-content">${candidate.content.substring(0, 200)}${candidate.content.length > 200 ? '...' : ''}</p>
        </div>
      `).join('')}
    </div>
  `;

  resultsElement.innerHTML = resultsHTML;
}

function clearFilter() {
  filteredCandidateIds = null;
  renderCandidatesList(allCandidates);
  document.getElementById('aiSearchResults').innerHTML = '<p class="placeholder-text">Enter a search query to find candidates using AI</p>';
  document.getElementById('aiSearchInput').value = '';
}

// Panel toggle event listeners - add at initialization
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      switchPanel(btn.dataset.panel);
    });
  });

  // AI Search button event listener
  document.getElementById('aiSearchBtn').addEventListener('click', performAISearch);

  // AI Search input - submit on Enter (with Shift+Enter for new line)
  document.getElementById('aiSearchInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      performAISearch();
    }
  });
});

// Set focus to search box on page load
setTimeout(() => {
  document.getElementById('searchBox')?.focus();
}, 100);

