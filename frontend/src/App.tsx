import { Link, Outlet } from "react-router-dom";

export default function App() {
  return (
    <div className="shell">
      <header className="topbar">
        <Link to="/" className="brand">
          PlanStride
        </Link>
        <span className="tagline">Turn floor plans into spaces you can explore</span>
      </header>
      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
