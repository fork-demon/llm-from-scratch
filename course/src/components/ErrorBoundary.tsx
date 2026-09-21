// One broken interactive must not blank the whole page. Every lab and diagram is wrapped
// in one of these, so a crash costs the learner that one box and nothing else.
import { Component, type ErrorInfo, type ReactNode } from 'react'
import { REPO_URL } from '../data/curriculum'

interface Props { children: ReactNode; what?: string }
interface State { error: Error | null }

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Keep it in the console so a learner can paste it into an issue.
    console.error('Interactive crashed:', this.props.what ?? 'unknown', error, info.componentStack)
  }

  render() {
    const { error } = this.state
    if (!error) return this.props.children
    const title = `Crash in ${this.props.what ?? 'an interactive'}`
    const body = `What I was doing:\n\n\nError: ${error.message}\nPage: ${window.location.hash}`
    const issue = `${REPO_URL}/issues/new?title=${encodeURIComponent(title)}&body=${encodeURIComponent(body)}`
    return (
      <div className="crash" role="alert">
        <p><b>This part of the page stopped working.</b> The rest of the lesson is fine, and your progress is safe.</p>
        <p className="muted" style={{ fontSize: 14 }}>{this.props.what ? `${this.props.what}: ` : ''}{error.message}</p>
        <div className="btn-row">
          <button className="btn small" onClick={() => this.setState({ error: null })}>Try again</button>
          <button className="btn small" onClick={() => window.location.reload()}>Reload the page</button>
          <a className="btn small" href={issue} target="_blank" rel="noreferrer">Report it</a>
        </div>
      </div>
    )
  }
}
