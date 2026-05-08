import { Component } from "react";

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, errorMessage: "" };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, errorMessage: error?.message || "Unknown error" };
  }

  componentDidCatch(error, errorInfo) {
    // Keep this for debugging in browser devtools.
    console.error("UI crash captured by ErrorBoundary", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <main className="page">
          <section className="card wide">
            <h2>UI Error</h2>
            <p>The app hit a runtime issue, so we paused rendering safely.</p>
            <p><strong>Details:</strong> {this.state.errorMessage}</p>
          </section>
        </main>
      );
    }

    return this.props.children;
  }
}
