import React, { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'

function Message({ msg, onFeedback }) {
  const [feedbackGiven, setFeedbackGiven] = useState(false)

  const handleFeedback = (value) => {
    if (feedbackGiven) return
    setFeedbackGiven(true)
    onFeedback(msg.conversationId, value)
  }

  const isUser = msg.role === 'user'

  return (
    <div className={`message ${isUser ? 'message-user' : 'message-assistant'}`}>
      <div className="message-avatar">
        {isUser ? '👤' : '🤖'}
      </div>
      <div className="message-content">
        <div className="message-sender">
          {isUser ? 'You' : 'FastAPI Assistant'}
        </div>
        {isUser ? (
          <div className="message-text">{msg.text}</div>
        ) : (
          <div className="message-text markdown-body">
            <ReactMarkdown>{msg.text}</ReactMarkdown>
          </div>
        )}
        {!isUser && msg.citations && msg.citations.length > 0 && (
          <div className="message-citations">
            <span className="citations-label">Sources:</span>
            {msg.citations.map((cite, i) => (
              <a
                key={i}
                href={cite.url}
                target="_blank"
                rel="noopener noreferrer"
                className="citation-link"
              >
                {cite.section || cite.title}
              </a>
            ))}
          </div>
        )}
        {!isUser && msg.relevance && (
          <div className={`message-relevance relevance-${msg.relevance.toLowerCase()}`}>
            {msg.relevance === 'RELEVANT' && '✅ '}
            {msg.relevance === 'PARTLY_RELEVANT' && '⚠️ '}
            {msg.relevance === 'NON_RELEVANT' && '❌ '}
            {msg.relevance}
          </div>
        )}
        {!isUser && (
          <div className="message-feedback">
            <button
              className={`feedback-btn ${feedbackGiven && msg.feedbackValue === 1 ? 'active' : ''}`}
              onClick={() => handleFeedback(1)}
              title="Helpful"
              disabled={feedbackGiven}
            >
              👍
            </button>
            <button
              className={`feedback-btn ${feedbackGiven && msg.feedbackValue === -1 ? 'active' : ''}`}
              onClick={() => handleFeedback(-1)}
              title="Not helpful"
              disabled={feedbackGiven}
            >
              👎
            </button>
          </div>
        )}
        {!isUser && msg.responseTime && (
          <div className="message-meta">
            {(msg.responseTime).toFixed(2)}s · {msg.tokenUsage?.total_tokens || 0} tokens
          </div>
        )}
      </div>
    </div>
  )
}

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSubmit = async (e) => {
    e.preventDefault()
    const question = input.trim()
    if (!question || loading) return

    setInput('')
    setError(null)

    // Add user message
    setMessages((prev) => [
      ...prev,
      { role: 'user', text: question },
    ])

    setLoading(true)

    try {
      const response = await fetch('/question', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })

      if (!response.ok) {
        const errData = await response.json()
        throw new Error(errData.detail || `HTTP ${response.status}`)
      }

      const data = await response.json()

      // Add assistant message
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          text: data.answer,
          conversationId: data.conversation_id,
          relevance: data.relevance,
          responseTime: data.response_time,
          tokenUsage: data.token_usage,
          citations: (data.retrieved_chunks || []).slice(0, 3),
          feedbackValue: null,
        },
      ])
    } catch (err) {
      setError(err.message)
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          text: `❌ Error: ${err.message}\n\nPlease make sure the backend server is running and try again.`,
          conversationId: null,
          relevance: null,
          responseTime: null,
          tokenUsage: null,
          citations: [],
          feedbackValue: null,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleFeedback = async (conversationId, value) => {
    if (!conversationId) return

    try {
      const response = await fetch('/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          conversation_id: conversationId,
          feedback: value,
        }),
      })

      if (!response.ok) {
        console.error('Feedback failed:', await response.text())
      }
    } catch (err) {
      console.error('Feedback error:', err)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="header-icon">🔍</div>
          <div className="header-text">
            <h1>FastAPI Docs Assistant</h1>
            <p className="header-subtitle">
              Ask questions about FastAPI — answers grounded in the official documentation
            </p>
          </div>
        </div>
      </header>

      <main className="chat-container">
        <div className="messages">
          {messages.length === 0 && !loading && (
            <div className="welcome">
              <div className="welcome-icon">💬</div>
              <h2>Ask me anything about FastAPI</h2>
              <p className="welcome-text">
                I can help you with path parameters, query parameters, request bodies,
                dependencies, CORS, middleware, error handling, testing, and more!
              </p>
              <div className="suggestions">
                <button
                  className="suggestion-btn"
                  onClick={() => setInput('How do I add CORS to my FastAPI app?')}
                >
                  How do I add CORS?
                </button>
                <button
                  className="suggestion-btn"
                  onClick={() => setInput('What is dependency injection in FastAPI?')}
                >
                  What is dependency injection?
                </button>
                <button
                  className="suggestion-btn"
                  onClick={() => setInput('How do I use path parameters with types?')}
                >
                  Path parameters with types
                </button>
                <button
                  className="suggestion-btn"
                  onClick={() => setInput('How do I test a FastAPI application?')}
                >
                  How to test FastAPI?
                </button>
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <Message key={i} msg={msg} onFeedback={handleFeedback} />
          ))}

          {loading && (
            <div className="message message-assistant">
              <div className="message-avatar">🤖</div>
              <div className="message-content">
                <div className="message-sender">FastAPI Assistant</div>
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}

          {error && (
            <div className="error-banner">
              ⚠️ {error}
              <button className="error-dismiss" onClick={() => setError(null)}>✕</button>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <form className="input-form" onSubmit={handleSubmit}>
          <input
            type="text"
            className="input-field"
            placeholder="Ask a question about FastAPI..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
          />
          <button
            type="submit"
            className="submit-btn"
            disabled={!input.trim() || loading}
          >
            {loading ? '⏳' : '➤'}
          </button>
        </form>
      </main>
    </div>
  )
}

export default App