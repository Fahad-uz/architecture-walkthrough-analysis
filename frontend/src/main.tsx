import React from "react";
import ReactDOM from "react-dom/client";
import { HashRouter, Route, Routes } from "react-router-dom";
import App from "./App";
import EditorPage from "./pages/EditorPage";
import PreviewPage from "./pages/PreviewPage";
import UploadPage from "./pages/UploadPage";
import WalkthroughPage from "./pages/WalkthroughPage";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <HashRouter>
      <Routes>
        <Route element={<App />}>
          <Route index element={<UploadPage />} />
          <Route path="jobs/:jobId/edit" element={<EditorPage />} />
          <Route path="jobs/:jobId/preview" element={<PreviewPage />} />
        </Route>
        <Route path="jobs/:jobId/walkthrough" element={<WalkthroughPage />} />
      </Routes>
    </HashRouter>
  </React.StrictMode>,
);
