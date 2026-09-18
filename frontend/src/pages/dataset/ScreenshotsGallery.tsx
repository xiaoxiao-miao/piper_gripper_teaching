import React, { useEffect, useState } from "react";

interface ScreenshotItem {
  filename: string;
  cam_id: string;
  mode: string;
  ts: number;
  url: string;
  color_files?: Record<string, string>; // 편광 카메라: {rgb,aolp,dolp,grid,stokes} -> 파일명
  raw_files?: Record<string, string>;   // 편광 카메라: {mosaic,aolp,dolp} -> 파일명 (raw, 비컬러라이즈)
}

const COLOR_MODE_ORDER = ["rgb", "aolp", "dolp", "grid", "stokes"] as const;

export default function ScreenshotsGallery() {
  const [items, setItems] = useState<ScreenshotItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [viewerItem, setViewerItem] = useState<ScreenshotItem | null>(null);
  const [viewerMode, setViewerMode] = useState<string>("rgb");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [downloading, setDownloading] = useState(false);

  const load = () => {
    setLoading(true);
    fetch("/api/screenshots")
      .then(r => r.json())
      .then((data: ScreenshotItem[]) => setItems(Array.isArray(data) ? data : []))
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  const handleDelete = (filename: string) => {
    fetch(`/api/screenshots/${encodeURIComponent(filename)}`, { method: "DELETE" })
      .then(() => {
        setConfirmDelete(null);
        setSelected(prev => { const next = new Set(prev); next.delete(filename); return next; });
        load();
      });
  };

  const toggleSelect = (filename: string) => {
    setSelected(prev => {
      const next = new Set(prev);
      if (next.has(filename)) next.delete(filename); else next.add(filename);
      return next;
    });
  };

  const handleDownloadSelected = () => {
    if (selected.size === 0 || downloading) return;
    setDownloading(true);
    fetch("/api/screenshots/download_raw12", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filenames: Array.from(selected) }),
    })
      .then(async r => {
        if (!r.ok) {
          const err = await r.json().catch(() => ({}));
          alert(err.error || "다운로드 실패");
          return;
        }
        const blob = await r.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "polarization_raw_bundle.zip";
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      })
      .finally(() => setDownloading(false));
  };

  const openViewer = (it: ScreenshotItem) => {
    setViewerItem(it);
    setViewerMode(it.mode || "rgb");
  };

  const viewerUrl = viewerItem
    ? `/api/screenshots/${viewerItem.color_files?.[viewerMode] ?? viewerItem.filename}`
    : "";

  return (
    <div style={styles.root}>
      <div style={styles.header}>
        <h2 style={styles.title}>Screenshots ({items.length})</h2>
        {selected.size > 0 && (
          <button onClick={handleDownloadSelected} disabled={downloading} style={styles.downloadBtn}>
            {downloading ? "묶는 중…" : `⬇ Raw Stokes(S0/S1/S2) 다운로드 (${selected.size})`}
          </button>
        )}
      </div>
      {loading ? (
        <div style={styles.empty}>Loading…</div>
      ) : items.length === 0 ? (
        <div style={styles.empty}>No screenshots yet — use the Screenshot button on the Capture page.</div>
      ) : (
        <div style={styles.grid}>
          {items.map(it => (
            <div key={it.filename} style={styles.card}>
              <div style={{ position: "relative" }}>
                <img
                  src={it.url}
                  alt={it.filename}
                  style={{ ...styles.thumb, cursor: "pointer" }}
                  onClick={() => openViewer(it)}
                />
                {it.raw_files && (
                  <label style={styles.selectCheckboxWrap} onClick={e => e.stopPropagation()}>
                    <input
                      type="checkbox"
                      checked={selected.has(it.filename)}
                      onChange={() => toggleSelect(it.filename)}
                    />
                  </label>
                )}
              </div>
              <div style={styles.meta}>
                <span>{it.cam_id}{it.mode ? ` · ${it.mode}` : ""}</span>
                <span>{it.ts ? new Date(it.ts * 1000).toLocaleString() : ""}</span>
              </div>
              {it.raw_files && (
                <div style={styles.rawRow}>
                  {Object.entries(it.raw_files).map(([key, fname]) => (
                    <a key={key} href={`/api/screenshots/${fname}`} target="_blank" rel="noreferrer" style={styles.rawLink}>
                      {key}.png
                    </a>
                  ))}
                </div>
              )}
              {confirmDelete === it.filename ? (
                <div style={styles.confirmRow}>
                  <button onClick={() => handleDelete(it.filename)} style={styles.deleteBtnConfirm}>Delete</button>
                  <button onClick={() => setConfirmDelete(null)} style={styles.cancelBtn}>Cancel</button>
                </div>
              ) : (
                <button onClick={() => setConfirmDelete(it.filename)} style={styles.deleteBtn}>Delete</button>
              )}
            </div>
          ))}
        </div>
      )}

      {viewerItem && (
        <div style={styles.viewerOverlay} onClick={() => setViewerItem(null)}>
          <div style={styles.viewerBox} onClick={e => e.stopPropagation()}>
            <img src={viewerUrl} alt={viewerItem.filename} style={styles.viewerImg} />
            <div style={styles.viewerBar}>
              {viewerItem.color_files ? (
                <div style={styles.viewerModes}>
                  {COLOR_MODE_ORDER.filter(m => viewerItem.color_files?.[m]).map(m => (
                    <button
                      key={m}
                      onClick={() => setViewerMode(m)}
                      style={{
                        ...styles.viewerModeBtn,
                        background: viewerMode === m ? "#0891b2" : "#f1f5f9",
                        color: viewerMode === m ? "#fff" : "#374151",
                      }}
                    >
                      {m === "grid" ? "2×2 Grid" : m === "stokes" ? "S1/S2" : m.toUpperCase()}
                    </button>
                  ))}
                </div>
              ) : <span />}
              <button onClick={() => setViewerItem(null)} style={styles.viewerCloseBtn}>✕ Close</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  root: { padding: 20, height: "100%", overflow: "auto", background: "#f8fafc" },
  header: { display: "flex", alignItems: "center", gap: 16, marginBottom: 16 },
  title: { fontSize: 18, fontWeight: 700, color: "#1e293b", margin: 0 },
  downloadBtn: {
    border: "none", borderRadius: 6, padding: "6px 14px",
    background: "#0891b2", color: "#fff", fontSize: 12, fontWeight: 700, cursor: "pointer",
  },
  selectCheckboxWrap: {
    position: "absolute", top: 6, left: 6,
    width: 22, height: 22, borderRadius: 5,
    background: "rgba(15,23,42,0.55)",
    display: "flex", alignItems: "center", justifyContent: "center",
    cursor: "pointer",
  },
  empty: { color: "#94a3b8", fontSize: 14, textAlign: "center", padding: 60 },
  grid: { display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 14 },
  card: { background: "#fff", border: "1px solid #e2e8f0", borderRadius: 10, overflow: "hidden" },
  thumb: { width: "100%", aspectRatio: "4/3", objectFit: "cover", display: "block", background: "#0f172a" },
  meta: { display: "flex", justifyContent: "space-between", padding: "8px 10px", fontSize: 11, color: "#64748b" },
  rawRow: { display: "flex", gap: 8, padding: "0 10px 8px", flexWrap: "wrap" },
  rawLink: { fontSize: 10, color: "#0891b2", textDecoration: "none", fontWeight: 700 },
  deleteBtn: {
    width: "100%", padding: 6, border: "none", borderTop: "1px solid #f1f5f9",
    background: "#fff", color: "#ef4444", fontSize: 11, fontWeight: 600, cursor: "pointer",
  },
  confirmRow: { display: "flex", borderTop: "1px solid #f1f5f9" },
  deleteBtnConfirm: {
    flex: 1, padding: 6, border: "none", background: "#fee2e2",
    color: "#dc2626", fontSize: 11, fontWeight: 700, cursor: "pointer",
  },
  cancelBtn: {
    flex: 1, padding: 6, border: "none", background: "#f1f5f9",
    color: "#374151", fontSize: 11, fontWeight: 600, cursor: "pointer",
  },
  viewerOverlay: {
    position: "fixed", inset: 0, zIndex: 1000,
    background: "rgba(0,0,0,0.85)",
    display: "flex", alignItems: "center", justifyContent: "center",
    padding: 24,
  },
  viewerBox: {
    display: "flex", flexDirection: "column", gap: 10,
    maxWidth: "90vw", maxHeight: "90vh",
  },
  viewerImg: {
    maxWidth: "90vw", maxHeight: "78vh", objectFit: "contain",
    borderRadius: 8, background: "#000",
  },
  viewerBar: {
    display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12,
  },
  viewerModes: { display: "flex", gap: 6 },
  viewerModeBtn: {
    border: "1px solid #e2e8f0", borderRadius: 6, padding: "6px 12px",
    fontSize: 12, fontWeight: 700, cursor: "pointer",
  },
  viewerCloseBtn: {
    border: "1px solid rgba(255,255,255,0.3)", borderRadius: 6, padding: "6px 12px",
    fontSize: 12, fontWeight: 700, cursor: "pointer", background: "rgba(15,23,42,0.7)", color: "#fff",
  },
};
