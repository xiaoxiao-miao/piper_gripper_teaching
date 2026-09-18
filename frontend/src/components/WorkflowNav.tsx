import React from "react";
import type { Page } from "../App";

interface Props {
  page: Page;
  onNavigate: (p: Page) => void;
}

const PAGE_LINKS: { page: Page; label: string }[] = [
  { page: "capture", label: "Capture" },
  { page: "dataset", label: "Dataset" },
  { page: "screenshots", label: "📷 Screenshots" },
  { page: "setup", label: "System Setup" },
  { page: "diagnostics", label: "Diagnostics" },
];

export default function WorkflowNav({ page, onNavigate }: Props) {
  return (
    <nav style={styles.nav}>
      <div style={styles.brand}>Piper Capture</div>
      <div style={styles.section}>
        <div style={styles.sectionTitle}>PAGES</div>
        {PAGE_LINKS.map(({ page: p, label }) => (
          <button
            key={p}
            style={{
              ...styles.navBtn,
              background: page === p ? "#e0e7ff" : "transparent",
              color: page === p ? "#3730a3" : "#374151",
              fontWeight: page === p ? 800 : 600,
            }}
            onClick={() => onNavigate(p)}
          >
            {label}
          </button>
        ))}
      </div>
    </nav>
  );
}

const styles: Record<string, React.CSSProperties> = {
  nav: {
    width: 160,
    background: "#fff",
    borderRight: "1px solid #e5e7eb",
    padding: "16px 8px",
    display: "flex",
    flexDirection: "column",
    gap: 18,
    flexShrink: 0,
    overflowY: "auto",
  },
  brand: {
    padding: "0 8px 10px",
    borderBottom: "1px solid #eef2f7",
    color: "#1e293b",
    fontSize: 14,
    fontWeight: 900,
  },
  section: {
    display: "flex",
    flexDirection: "column",
    gap: 2,
  },
  sectionTitle: {
    fontSize: 10,
    fontWeight: 700,
    color: "#94a3b8",
    letterSpacing: 1,
    padding: "4px 8px",
    marginBottom: 4,
  },
  navBtn: {
    display: "block",
    width: "100%",
    textAlign: "left",
    border: "none",
    borderRadius: 6,
    padding: "7px 12px",
    fontSize: 13,
    cursor: "pointer",
  },
};
