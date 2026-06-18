import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App'

// Inject API key from meta tag (inserted by backend when API_KEY env is set)
const apiKey = document.querySelector('meta[name="api-key"]')?.getAttribute('content') || ''
if (apiKey) {
  const orig = window.fetch.bind(window)
  window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
    const headers = new Headers(init?.headers)
    headers.set('x-api-key', apiKey)
    return orig(input, { ...init, headers })
  }
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
