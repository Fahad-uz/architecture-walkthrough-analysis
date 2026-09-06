import React from "react";
import ReactDOM from "react-dom/client";
import { HashRouter, Route, Routes } from "react-router-dom";
import App from "./App";
import UploadPage from "./pages/UploadPage";
import "./index.css";

const EditorPage = React.lazy(() => import("./pages/EditorPage"));
const PreviewPage = React.lazy(() => import("./pages/PreviewPage"));
const WalkthroughPage = React.lazy(() => import("./pages/WalkthroughPage"));

function RouteLoader({ label = "Loading workspace…" }: { label?: string }) {
  return (
    <div className="page-status" role="status" aria-live="polite">
      {label}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <HashRouter>
      <Routes>
        <Route element={<App />}>
          <Route index element={<UploadPage />} />
          <Route
            path="jobs/:jobId/edit"
            element={
              <React.Suspense fallback={<RouteLoader label="Loading editor…" />}>
                <EditorPage />
              </React.Suspense>
            }
          />
          <Route
            path="jobs/:jobId/preview"
            element={
              <React.Suspense fallback={<RouteLoader label="Loading preview…" />}>
                <PreviewPage />
              </React.Suspense>
            }
          />
        </Route>
        <Route
          path="jobs/:jobId/walkthrough"
          element={
            <React.Suspense fallback={<RouteLoader label="Loading walkthrough…" />}>
              <WalkthroughPage />
            </React.Suspense>
          }
        />
      </Routes>
    </HashRouter>
  </React.StrictMode>,
);
