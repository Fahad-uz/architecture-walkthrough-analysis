import { Link, Outlet } from "react-router-dom";

export default function App() {
  return (
    <div className="shell">
      <header className="topbar">
        <Link to="/" className="brand">
          Architecture Walkthrough
        </Link>
        <span className="tagline">plan image → editable layout → lit 3D walkthrough</span>
      </header>
      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
