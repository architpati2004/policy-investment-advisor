import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { AlertsPage } from "./pages/AlertsPage";
import { ChatPage } from "./pages/ChatPage";
import { DashboardPage } from "./pages/DashboardPage";
import { DocumentsPage } from "./pages/DocumentsPage";
import { PortfolioPage } from "./pages/PortfolioPage";

const NAV = [
  { to: "/chat", label: "Research chat" },
  { to: "/", label: "Dashboard", end: true },
  { to: "/alerts", label: "Alerts" },
  { to: "/portfolio", label: "Portfolio" },
  { to: "/documents", label: "Documents" },
];

export function App() {
  return (
    <div className="shell">
      <nav className="nav">
        <div className="brand">
          Policy-Based Investment Advisor
          <span className="brand-note">local · grounded · cited</span>
        </div>
        <ul>
          {NAV.map((item) => (
            <li key={item.to}>
              <NavLink to={item.to} end={item.end} className={({ isActive }) => (isActive ? "active" : "")}>
                {item.label}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
      <main>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/alerts" element={<AlertsPage />} />
          <Route path="/portfolio" element={<PortfolioPage />} />
          <Route path="/documents" element={<DocumentsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
