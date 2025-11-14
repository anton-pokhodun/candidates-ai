const API_BASE_URL = 'http://localhost:8000';
let allCandidates = [];
let selectedCandidateId = null;
let currentEventSource = null;

// ===== DOM Utilities =====
function showLoading(element, message = 'Loading...') {
  element.innerHTML = `<div class="loading"><div class="spinner"></div>${message}</div>`;
}

function showError(element, message) {
  element.innerHTML = `<div class="error">${message}</div>`;
}

function getElement(id) {
  return document.getElementById(id);
}

function formatMarkdown(text) {
  return text
    .replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/^- (.+)$/gm, '<li>$1</li>');
}

function formatAgentOutput(text) {
  return text
    .replace(/Thought:/g, '<br><span class="agent-thought">Thought:</span>')
    .replace(/Action:/g, '<br><span class="agent-action">Action:</span>')
    .replace(/Action Input:/g, '<br><span class="agent-action">Action Input:</span>');
}

function wrapListItems(html) {
  return html.replace(/(<li>.*?<\/li>\s*)+/g, (match) => {
    return '<ul>' + match + '</ul>';
  });
}

function formatContent(content) {
  let formatted = formatMarkdown(content);
  formatted = formatAgentOutput(formatted);
  formatted = wrapListItems(formatted);
  return `<p>${formatted}</p>`;
}

// ===== EventSource Management =====
function closeCurrentEventSource() {
  if (currentEventSource) {
    currentEventSource.close();
    currentEventSource = null;
  }
}

function createEventSource(url, handlers) {
  closeCurrentEventSource();

  const eventSource = new EventSource(url);
  currentEventSource = eventSource;

  eventSource.onmessage = handlers.onMessage;
  eventSource.onerror = (error) => {
    console.error('EventSource error:', error);
    eventSource.close();
    currentEventSource = null;
    if (handlers.onError) {
      handlers.onError(error);
    }
  };

  return eventSource;
}

// ===== UI State Management =====
function setActiveState(items, activeId, dataAttribute = 'id') {
  items.forEach(item => {
    item.classList.remove('active');
    if (item.dataset[dataAttribute] === activeId) {
      item.classList.add('active');
    }
  });
}

function toggleButton(button, disabled, text) {
  button.disabled = disabled;
  button.textContent = text;
}

// ===== Candidate Rendering =====
function renderCandidateItem(candidate) {
  return `
    <div class="candidate-item" onclick="selectCandidate('${candidate.candidate_id}')" data-id="${candidate.candidate_id}">
      <h3>${candidate.candidate_name}</h3>
      <p>#${candidate.candidate_id}</p>
      <p>file:${candidate.file_name}</p>
    </div>
  `;
}

function renderCandidatesList(candidates) {
  const listElement = getElement('candidatesList');

  if (candidates.length === 0) {
    listElement.innerHTML = '<p style="color: #999; text-align: center;">No candidates found</p>';
    return;
  }

  listElement.innerHTML = candidates.map(renderCandidateItem).join('');
}

function renderCandidateMetadata(metadata) {
  return `
    <div class="summary-content">
      <h1>${metadata.candidate_name}</h1>
      <p><strong>ID:</strong> ${metadata.candidate_id}</p>
      <p><strong>File:</strong> ${metadata.file_name}</p>
      <hr style="margin: 20px 0; border: 1px solid #e0e0e0;">
      <div id="streamingContent"></div>
    </div>
  `;
}

// ===== Search Result Rendering =====
function renderSearchResultItem(candidate, index) {
  return `
    <div class="search-result-item" onclick="selectCandidate('${candidate.candidate_id}')">
      <div class="result-header">
        <h4>${index + 1}. ${candidate.candidate_name}</h4>
        <span class="score-badge">${(candidate.score * 100).toFixed(1)}%</span>
      </div>
      <p class="result-meta">ID: ${candidate.candidate_id} | File: ${candidate.file_name}</p>
      <p class="result-content">${candidate.content.substring(0, 200)}${candidate.content.length > 200 ? '...' : ''}</p>
    </div>
  `;
}

function renderSearchResults(candidates) {
  if (!candidates || candidates.length === 0) return '';

  return `
    <button class="clear-filter-btn" onclick="clearFilter()">Clear Filter & Show All</button>
    ${candidates.map((candidate, idx) => renderSearchResultItem(candidate, idx)).join('')}
  `;
}

function initializeSearchResultsView() {
  return `
    <div class="search-summary">
      <h3>Search Results</h3>
      <div id="streamingAnswer"></div>
    </div>
    <div class="search-results-list" id="searchResultsList"></div>
  `;
}

// ===== API Calls =====
async function fetchCandidates() {
  const response = await fetch(`${API_BASE_URL}/candidates`);
  const data = await response.json();
  return data.candidates;
}

async function loadCandidates() {
  const listElement = getElement('candidatesList');
  showLoading(listElement, 'Loading candidates...');

  try {
    allCandidates = await fetchCandidates();
    renderCandidatesList(allCandidates);
  } catch (error) {
    showError(listElement, `Failed to load candidates: ${error.message}`);
  }
}

// ===== Streaming Content Updates =====
function updateStreamingContent(content) {
  const contentElement = getElement('streamingContent');
  if (!contentElement) return;

  contentElement.innerHTML = formatContent(content);
  contentElement.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

function updateStreamingAnswer(answer, candidates) {
  const answerElement = getElement('streamingAnswer');
  const resultsListElement = getElement('searchResultsList');

  if (answerElement) {
    answerElement.innerHTML = formatContent(answer);
  }

  if (resultsListElement && candidates && candidates.length > 0) {
    resultsListElement.innerHTML = renderSearchResults(candidates);
  }
}

// ===== Candidate Selection =====
async function selectCandidate(candidateId) {
  closeCurrentEventSource();
  selectedCandidateId = candidateId;

  const candidateItems = document.querySelectorAll('.candidate-item');
  setActiveState(candidateItems, candidateId, 'id');

  const detailsElement = getElement('candidate-details');
  detailsElement.classList.remove('empty');
  showLoading(detailsElement, 'Loading candidate details...');

  try {
    let metadata = null;
    let contentBuffer = '';

    createEventSource(
      `${API_BASE_URL}/candidates/${encodeURIComponent(candidateId)}`,
      {
        onMessage: (event) => {
          try {
            const message = JSON.parse(event.data);

            if (message.type === 'metadata') {
              metadata = message.data;
              detailsElement.innerHTML = renderCandidateMetadata(metadata);
            } else if (message.type === 'content') {
              contentBuffer += message.data;
              updateStreamingContent(contentBuffer);
            } else if (message.type === 'done') {
              closeCurrentEventSource();
            }
          } catch (error) {
            console.error('Error parsing SSE message:', error);
          }
        },
        onError: () => {
          if (!metadata) {
            showError(detailsElement, 'Failed to load candidate details. Please try again.');
          }
        }
      }
    );
  } catch (error) {
    showError(detailsElement, `Failed to load candidate details: ${error.message}`);
  }
}

// ===== Stream Reader =====
async function readStream(reader, decoder, onData) {
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value, { stream: true });
    const lines = chunk.split('\n');

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const message = JSON.parse(line.slice(6));
          onData(message);
        } catch (e) {
          console.error('Error parsing SSE message:', e);
        }
      }
    }
  }
}

// ===== AI Search =====
async function performAISearch() {
  const query = getElement('aiSearchInput').value.trim();
  if (!query) return;

  const resultsElement = getElement('aiSearchResults');
  const searchBtn = getElement('aiSearchBtn');

  toggleButton(searchBtn, true, 'Searching...');
  showLoading(resultsElement, 'Searching candidates...');

  try {
    const response = await fetch(`${API_BASE_URL}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k: 10 })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let candidates = [];
    let answerBuffer = '';

    resultsElement.innerHTML = initializeSearchResultsView();

    await readStream(reader, decoder, (message) => {
      if (message.type === 'metadata') {
        candidates = message.data.candidates;
      } else if (message.type === 'content') {
        answerBuffer += message.data;
        updateStreamingAnswer(answerBuffer, candidates);
      }
    });

    toggleButton(searchBtn, false, 'Search');
  } catch (error) {
    console.error('Search error:', error);
    showError(resultsElement, `Search failed: ${error.message}`);
    toggleButton(searchBtn, false, 'Search');
  }
}

function clearFilter() {
  renderCandidatesList(allCandidates);
  getElement('aiSearchInput').innerHTML = '<p class="placeholder-text">Ask about candidates (e.g., "Find software engineers with Python experience", "Search Wikipedia for React", "Create a superhero based on candidate skills")</p>';
  // getElement('aiSearchInput').value = '';
}

// ===== Panel Management =====
function switchPanel(panelName) {
  const toggleButtons = document.querySelectorAll('.toggle-btn');
  setActiveState(toggleButtons, panelName, 'panel');

  document.querySelectorAll('.panel-content').forEach(panel => {
    panel.classList.remove('active');
  });

  const panelMap = {
    'details': 'candidate-details',
    'search': 'searchPanel'
  };

  const panelId = panelMap[panelName];
  if (panelId) {
    getElement(panelId).classList.add('active');
  }
}

// ===== Event Handlers =====
function initializeSearchBox() {
  getElement('searchBox').addEventListener('input', (e) => {
    const searchTerm = e.target.value.toLowerCase();
    const filtered = allCandidates.filter(candidate =>
      candidate.candidate_name.toLowerCase().includes(searchTerm)
    );
    renderCandidatesList(filtered);
  });
}

function initializePanelToggles() {
  document.querySelectorAll('.toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      switchPanel(btn.dataset.panel);
    });
  });
}

function initializeAISearch() {
  getElement('aiSearchBtn').addEventListener('click', performAISearch);

  getElement('aiSearchInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      performAISearch();
    }
  });
}

// ===== Initialization =====
document.addEventListener('DOMContentLoaded', () => {
  initializePanelToggles();
  initializeAISearch();
});

window.addEventListener('beforeunload', () => {
  closeCurrentEventSource();
});

loadCandidates();
initializeSearchBox();

setTimeout(() => {
  getElement('searchBox')?.focus();
}, 100);
