import React from "react";
import type { Episode, ReviewIssue } from "../../types";
import { COMPLETENESS, fmtDate, fmtDuration, fmtMb, statusLabel, type ViewMode } from "./datasetUtils";

export function CompletenessChips({ ep }: { ep: Episode }) {
  const c = ep.completeness ?? {};
  return (
    <div style={styles.chipRow}>
      {COMPLETENESS.map(item => {
        const ok = !!c[item.key];
        return (
          <span
            key={item.key}
            title={`${item.title}: ${ok ? "available" : "missing"}`}
            style={{
              ...styles.dataChip,
              background: ok ? "#ecfdf5" : "#f8fafc",
              color: ok ? "#15803d" : "#94a3b8",
              borderColor: ok ? "#bbf7d0" : "#e2e8f0",
            }}
          >
            {item.label}
          </span>
        );
      })}
    </div>
  );
}

export function ReviewIssueBadges({ issues, compact = false }: { issues?: ReviewIssue[]; compact?: boolean }) {
  if (!issues?.length) return null;
  const visible = compact ? issues.slice(0, 2) : issues;
  return (
    <div style={styles.issueRow}>
      {visible.map(issue => (
        <span
          key={`${issue.key}-${issue.lerobot_index ?? ""}`}
          title={[issue.label, issue.note, issue.status].filter(Boolean).join(" · ")}
          style={{
            ...styles.issueBadge,
            background: issue.severity === "danger" ? "#fee2e2" : issue.severity === "warning" ? "#fff7ed" : "#eef2ff",
            color: issue.severity === "danger" ? "#b91c1c" : issue.severity === "warning" ? "#c2410c" : "#3730a3",
            borderColor: issue.severity === "danger" ? "#fecaca" : issue.severity === "warning" ? "#fed7aa" : "#c7d2fe",
          }}
        >
          {issue.label}
        </span>
      ))}
      {compact && issues.length > visible.length && (
        <span style={{ ...styles.issueBadge, background: "#f1f5f9", color: "#64748b", borderColor: "#e2e8f0" }}>
          +{issues.length - visible.length}
        </span>
      )}
    </div>
  );
}

function VideoThumb({
  src, active, label, success, style,
}: { src?: string | null; active: boolean; label: string; success?: boolean | null; style?: React.CSSProperties }) {
  const [visible, setVisible] = React.useState(false);
  const ref = React.useRef<HTMLDivElement>(null);
  const videoRef = React.useRef<HTMLVideoElement>(null);

  React.useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const obs = new IntersectionObserver(entries => {
      setVisible(entries.some(e => e.isIntersecting));
    }, { rootMargin: "180px" });
    obs.observe(node);
    return () => obs.disconnect();
  }, []);

  React.useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    if (active && visible) {
      video.play().catch(() => {});
    } else {
      video.pause();
      video.currentTime = 0;
    }
  }, [active, visible, src]);

  return (
    <div ref={ref} style={{ ...styles.videoThumb, ...style }}>
      {src && visible ? (
        <video ref={videoRef} muted loop playsInline preload="metadata" src={src} style={styles.videoThumbMedia} aria-label={label} />
      ) : (
        <div style={styles.videoThumbEmpty}>{src ? "Preview" : "No video"}</div>
      )}
      {success !== undefined && (
        <span style={{
          ...styles.thumbStatusOverlay,
          background: success === true ? "rgba(22,163,74,0.88)" : success === false ? "rgba(239,68,68,0.88)" : "rgba(100,116,139,0.88)",
        }}>
          {success === true ? "OK" : success === false ? "FAIL" : "NEW"}
        </span>
      )}
    </div>
  );
}

export function StatusBadge({ success, long = false }: { success: boolean | null | undefined; long?: boolean }) {
  const label = long ? statusLabel(success) : success === true ? "✓" : success === false ? "✗" : "?";
  if (success === null || success === undefined)
    return <span style={{ ...styles.badge, background: "#f1f5f9", color: "#6b7280" }}>{label}</span>;
  if (success)
    return <span style={{ ...styles.badge, background: "#dcfce7", color: "#15803d" }}>{label}</span>;
  return <span style={{ ...styles.badge, background: "#fee2e2", color: "#dc2626" }}>{label}</span>;
}

export function EpisodeCollectionItem({
  ep, index, viewMode, inspected, selected, hovered, onClick, onCheck, onHover, onDelete,
}: {
  ep: Episode;
  index: number;
  viewMode: ViewMode;
  inspected: boolean;
  selected: boolean;
  hovered: boolean;
  onClick: (ep: Episode, index: number, e: React.MouseEvent) => void;
  onCheck: (epId: string, index: number, e: React.MouseEvent) => void;
  onHover: (epId: string | null) => void;
  onDelete: (epId: string, e: React.MouseEvent) => void;
}) {
  const activePreview = hovered || inspected || selected;
  if (viewMode === "compact") {
    return (
      <div style={{ ...styles.compactRow, background: inspected ? "#eef2ff" : "#fff", borderColor: selected ? "#6366f1" : "#e5e7eb" }}
        onClick={e => onClick(ep, index, e)} onMouseEnter={() => onHover(ep.episode_id)} onMouseLeave={() => onHover(null)}>
        <input type="checkbox" checked={selected} onChange={() => {}} onClick={e => onCheck(ep.episode_id, index, e)} style={styles.checkbox} />
        <span style={styles.compactId}>{ep.episode_id.slice(-18)}</span>
        <span style={styles.compactTask}>{ep.task || "unspecified"}</span>
        <span><StatusBadge success={ep.success} long /></span>
        <span style={styles.compactMeta}>{ep.takes_count ?? 0}t</span>
        <span style={styles.compactMeta}>{fmtDuration(ep.duration_s ?? undefined)}</span>
        <span style={styles.compactMeta}>{fmtMb(ep.size_mb)}</span>
        {ep.review_issue_count ? <span style={styles.issueCount}>Issues {ep.review_issue_count}</span> : null}
      </div>
    );
  }

  if (viewMode === "grid") {
    return (
      <div style={{ ...styles.gridCard, borderColor: selected ? "#6366f1" : inspected ? "#a5b4fc" : "#e5e7eb", background: inspected ? "#f8faff" : "#fff" }}
        onClick={e => onClick(ep, index, e)} onMouseEnter={() => onHover(ep.episode_id)} onMouseLeave={() => onHover(null)}>
        <input type="checkbox" checked={selected} onChange={() => {}} onClick={e => onCheck(ep.episode_id, index, e)} style={styles.gridCheckbox} />
        <button onClick={e => onDelete(ep.episode_id, e)} style={styles.gridDeleteBtn} title="Delete episode">×</button>
        <VideoThumb src={ep.preview_video_url} active={activePreview} label={ep.episode_id} success={ep.success} style={{ width: "100%" }} />
        <div style={styles.gridTask}>{ep.task || "unspecified"}</div>
        <div style={styles.gridMeta}>{ep.takes_count ?? 0}t · {fmtDuration(ep.duration_s ?? undefined)} · {fmtMb(ep.size_mb)}</div>
        <ReviewIssueBadges issues={ep.review_issues} compact />
        <CompletenessChips ep={ep} />
      </div>
    );
  }

  return (
    <div style={{ ...styles.thumbRow, background: inspected ? "#eef2ff" : "#fff", borderColor: selected ? "#6366f1" : "#e5e7eb" }}
      onClick={e => onClick(ep, index, e)} onMouseEnter={() => onHover(ep.episode_id)} onMouseLeave={() => onHover(null)}>
      <input type="checkbox" checked={selected} onChange={() => {}} onClick={e => onCheck(ep.episode_id, index, e)} style={styles.checkbox} />
      <VideoThumb src={ep.preview_video_url} active={activePreview} label={ep.episode_id} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={styles.thumbTitleRow}>
          <span style={styles.epId}>{ep.episode_id.slice(-22)}</span>
          {ep.has_postprocess && <span title="postprocess edit" style={styles.postprocessBadge}>Export edit</span>}
          <StatusBadge success={ep.success} long />
        </div>
        <div style={styles.thumbTask}>{ep.task || "unspecified"}</div>
        <div style={styles.thumbMeta}>
          {ep.takes_count ?? 0} take · {fmtDuration(ep.duration_s ?? undefined)} · {fmtMb(ep.size_mb)} · {fmtDate(ep.created_at)}
        </div>
        <ReviewIssueBadges issues={ep.review_issues} compact />
        <CompletenessChips ep={ep} />
      </div>
      <button onClick={e => onDelete(ep.episode_id, e)} style={styles.rowDeleteBtn} title="Delete episode">×</button>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  checkbox: { accentColor: "#6366f1", flexShrink: 0 },
  badge: { padding: "1px 6px", borderRadius: 4, fontSize: 10, fontWeight: 700 },
  epId: { fontFamily: "monospace", fontSize: 10, color: "#6366f1", fontWeight: 700 },
  thumbRow: {
    display: "flex", alignItems: "center", gap: 10,
    border: "1px solid #e5e7eb", borderRadius: 10,
    padding: 9, cursor: "pointer", minHeight: 98,
  },
  thumbTitleRow: { display: "flex", alignItems: "center", gap: 6, minWidth: 0 },
  thumbTask: { fontSize: 13, color: "#111827", fontWeight: 700, marginTop: 5, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" },
  thumbMeta: { fontSize: 11, color: "#64748b", marginTop: 4 },
  postprocessBadge: { fontSize: 9, background: "#dbeafe", color: "#1d4ed8", borderRadius: 3, padding: "1px 5px", fontWeight: 800 },
  issueRow: { display: "flex", gap: 4, flexWrap: "wrap", marginTop: 6 },
  issueBadge: {
    border: "1px solid #c7d2fe",
    borderRadius: 5,
    padding: "1px 5px",
    fontSize: 9,
    fontWeight: 900,
    lineHeight: 1.45,
  },
  issueCount: {
    gridColumn: "1 / -1",
    justifySelf: "start",
    borderRadius: 5,
    padding: "1px 5px",
    background: "#fff7ed",
    color: "#c2410c",
    fontSize: 9,
    fontWeight: 900,
  },
  rowDeleteBtn: { width: 24, height: 24, borderRadius: 5, border: "none", background: "#f8fafc", color: "#94a3b8", cursor: "pointer", fontSize: 16, lineHeight: 1, flexShrink: 0 },
  videoThumb: {
    position: "relative", width: 104, aspectRatio: "16 / 9", borderRadius: 8,
    overflow: "hidden", background: "#0f172a", flexShrink: 0,
    display: "flex", alignItems: "center", justifyContent: "center",
  },
  videoThumbMedia: { width: "100%", height: "100%", objectFit: "cover", display: "block" },
  videoThumbEmpty: { color: "#94a3b8", fontSize: 11, fontWeight: 700 },
  thumbStatusOverlay: { position: "absolute", top: 6, right: 6, color: "#fff", borderRadius: 4, padding: "1px 5px", fontSize: 9, fontWeight: 900, letterSpacing: 0.4, lineHeight: 1.4 },
  chipRow: { display: "flex", gap: 4, flexWrap: "wrap", marginTop: 6 },
  dataChip: { border: "1px solid #e2e8f0", borderRadius: 4, padding: "1px 4px", fontSize: 9, fontWeight: 800 },
  compactRow: {
    display: "grid", gridTemplateColumns: "22px minmax(0, 1.15fr) minmax(0, 1.45fr) 76px 44px 70px 78px",
    gap: 6, alignItems: "center", border: "1px solid #e5e7eb", borderRadius: 8,
    padding: "7px 8px", cursor: "pointer", boxSizing: "border-box", width: "100%", minWidth: 0,
  },
  compactId: { fontFamily: "monospace", fontSize: 10, color: "#4f46e5", fontWeight: 800, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" },
  compactTask: { fontSize: 11, color: "#111827", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" },
  compactMeta: { fontSize: 11, color: "#64748b" },
  gridCard: { position: "relative", border: "1px solid #e5e7eb", borderRadius: 10, background: "#fff", padding: 7, cursor: "pointer", minWidth: 0 },
  gridCheckbox: { position: "absolute", top: 8, left: 8, zIndex: 2, accentColor: "#6366f1" },
  gridDeleteBtn: {
    position: "absolute", top: 8, right: 8, zIndex: 2,
    width: 22, height: 22, border: "none", borderRadius: 5,
    background: "rgba(15,23,42,0.58)", color: "#fff", cursor: "pointer",
  },
  gridTask: { fontSize: 10, fontWeight: 800, color: "#111827", marginTop: 6, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" },
  gridMeta: { fontSize: 9, color: "#64748b", margin: "3px 0 4px" },
};
