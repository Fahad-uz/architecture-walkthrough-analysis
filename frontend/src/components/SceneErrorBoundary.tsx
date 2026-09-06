import { Component, type ErrorInfo, type ReactNode } from "react";

interface SceneErrorBoundaryProps {
  children: ReactNode;
  resetKey: string | null;
  fallback: ReactNode;
  onError?: (error: Error) => void;
}

interface SceneErrorBoundaryState {
  error: Error | null;
}

/** Keep a failed GLB or WebGL context from taking down the surrounding editor
 * controls. Changing the model URL resets the boundary for the next build. */
export default class SceneErrorBoundary extends Component<
  SceneErrorBoundaryProps,
  SceneErrorBoundaryState
> {
  state: SceneErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): SceneErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, _info: ErrorInfo): void {
    this.props.onError?.(error);
  }

  componentDidUpdate(previous: SceneErrorBoundaryProps): void {
    if (this.state.error && previous.resetKey !== this.props.resetKey) {
      this.setState({ error: null });
    }
  }

  render(): ReactNode {
    return this.state.error ? this.props.fallback : this.props.children;
  }
}
