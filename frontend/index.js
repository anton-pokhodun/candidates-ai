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
  const detailsElement = document.getElementById('candidateDetails');
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

