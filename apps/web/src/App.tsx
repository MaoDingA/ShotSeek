import {
  type ChangeEvent,
  type DragEvent,
  type FormEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  cancelJob,
  exportUrl,
  listenToJob,
  loadEvidence,
  loadWorkspace,
  mediaUrl,
  previewUrl,
  searchVideo,
  uploadVideo,
} from "./api";
import type {
  AgentTrace,
  EvidenceResponse,
  HealthResponse,
  JobEvent,
  JobRecord,
  SearchHit,
  VideoRecord,
} from "./types";
import { terminalStates } from "./types";

const stageLabels: Record<string, string> = {
  CREATED: "创建任务",
  QUEUED: "等待处理",
  PROBING: "读取媒体",
  TRANSCODING: "生成代理",
  EXTRACTING_AUDIO: "提取音频",
  DETECTING_SHOTS: "检测镜头",
  CHUNKING: "切分片段",
  ANALYZING_VISUAL: "理解画面",
  ANALYZING_ASR: "识别对白",
  ALIGNING: "对齐时间线",
  BUILDING_SCENES: "构建场景",
  INDEXING: "建立索引",
  RETRYING: "正在重试",
  READY: "可以搜索",
  PARTIAL: "部分可用",
  FAILED: "处理失败",
  CANCELLED: "已取消",
};

function Icon({ name, size = 18 }: { name: string; size?: number }) {
  const paths: Record<string, React.ReactNode> = {
    search: <><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></>,
    upload: <><path d="M12 16V4"/><path d="m7 9 5-5 5 5"/><path d="M5 20h14"/></>,
    play: <path d="m8 5 11 7-11 7Z"/>,
    evidence: <><path d="M4 5h16v14H4z"/><path d="m7 15 3-3 2 2 3-4 2 3"/></>,
    trace: <><circle cx="5" cy="6" r="2"/><circle cx="19" cy="6" r="2"/><circle cx="12" cy="18" r="2"/><path d="M7 6h10M6.5 7.5l4.5 8.7M17.5 7.5 13 16.2"/></>,
    close: <><path d="m6 6 12 12M18 6 6 18"/></>,
    settings: <><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-2.8 2.8-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6v.2h-4V21a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1L4.2 17l.1-.1a1.7 1.7 0 0 0 .3-1.9A1.7 1.7 0 0 0 3 14H2.8v-4H3a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9L4.2 7 7 4.2l.1.1A1.7 1.7 0 0 0 9 4.6a1.7 1.7 0 0 0 1-1.6v-.2h4V3a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1L19.8 7l-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1h.2v4H21a1.7 1.7 0 0 0-1.6 1Z"/></>,
    chevron: <path d="m9 18 6-6-6-6"/>,
    check: <path d="m5 12 4 4L19 6"/>,
  };
  return <svg className="icon" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}

function timecode(ms: number | null | undefined): string {
  if (ms == null) return "--:--:--.---";
  const hours = Math.floor(ms / 3_600_000);
  const minutes = Math.floor((ms % 3_600_000) / 60_000);
  const seconds = Math.floor((ms % 60_000) / 1000);
  const millis = ms % 1000;
  return [hours, minutes, seconds].map((value) => String(value).padStart(2, "0")).join(":") + `.${String(millis).padStart(3, "0")}`;
}

function fileSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function noResultCopy(trace: AgentTrace | null): {
  heading: string;
  detail: string;
} {
  const rawCandidateCount = trace?.retrieval.candidate_count;
  const candidateCount = typeof rawCandidateCount === "number"
    ? rawCandidateCount
    : null;
  if (candidateCount === 0) {
    return {
      heading: "没有找到匹配的已索引描述",
      detail: "当前时间线没有记录与这句话对应的画面或对白标签。可以补充人物外观、动作或物体。",
    };
  }
  return {
    heading: "只找到部分相似画面",
    detail: "查询已经理解，但当前视频没有同时满足全部人物、动作和物体条件的直接证据。ShotSeek 不会用相似人物或画面冒充命中。",
  };
}

function StatusPill({ value }: { value: string }) {
  const tone = value === "LIVE" || value === "READY" ? "live" : value === "FAILED" ? "failed" : value === "CACHED" ? "cached" : "neutral";
  return <span className={`status-pill ${tone}`}><span className="status-dot" />{value}</span>;
}

function UploadSurface({ onFile, busy }: { onFile: (file: File) => void; busy: boolean }) {
  const [dragging, setDragging] = useState(false);
  const accept = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files[0];
    if (file) onFile(file);
  };
  return (
    <div className={`upload-surface ${dragging ? "dragging" : ""}`} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={accept}>
      <div className="upload-mark"><Icon name="upload" size={26} /></div>
      <div>
        <h2>{busy ? "正在接收视频" : "把长视频放到这里"}</h2>
        <p>MP4、MOV、MKV、AVI 或 WebM · 最大 4 GB</p>
      </div>
      <label className="primary-button upload-button">
        {busy ? "上传中…" : "选择视频"}
        <input type="file" accept="video/*,.mkv" disabled={busy} onChange={(event: ChangeEvent<HTMLInputElement>) => { const file = event.target.files?.[0]; if (file) onFile(file); }} />
      </label>
    </div>
  );
}

function ProcessingPanel({ job, onCancel }: { job: JobRecord; onCancel: () => void }) {
  const percent = Math.round(job.progress * 100);
  return (
    <section className="processing-panel">
      <div className="processing-copy">
        <span className="eyebrow">PRODUCTION RUNTIME</span>
        <h2>{stageLabels[job.state] ?? job.state}</h2>
        <p>{job.message}</p>
      </div>
      <div className="progress-value">{percent}<small>%</small></div>
      <div className="progress-track"><span style={{ width: `${percent}%` }} /></div>
      <div className="processing-meta">
        <span>{job.total_units ? `${job.completed_units} / ${job.total_units}` : "阶段初始化中"}</span>
        <span>重试 {job.retry_count}</span>
        {!terminalStates.includes(job.state) && <button className="text-button" onClick={onCancel}>取消任务</button>}
      </div>
    </section>
  );
}

function ResultCard({ hit, index, active, videoId, onOpen }: { hit: SearchHit; index: number; active: boolean; videoId: string; onOpen: () => void }) {
  const scene = hit.candidate;
  return (
    <button className={`result-card ${active ? "active" : ""}`} onClick={onOpen}>
      <div className="result-preview">
        <img src={previewUrl(videoId, scene.scene_id)} alt="场景关键帧" loading="lazy" />
        <span className="rank">{String(index + 1).padStart(2, "0")}</span>
        <span className="preview-time">{timecode(scene.start_ms)}</span>
      </div>
      <div className="result-body">
        <div className="result-heading">
          <strong>{scene.summary}</strong>
          <span>{Math.round(hit.final_score * 100)}%</span>
        </div>
        <p>{hit.verification.reason}</p>
        <div className="evidence-chips">
          {scene.evidence_refs.some((item) => item.kind === "visual") && <span>画面</span>}
          {scene.evidence_refs.some((item) => item.kind === "dialogue") && <span>对白</span>}
          {scene.shot_ids.length > 0 && <span>镜头边界</span>}
          {hit.verification.direct_evidence && <span className="verified"><Icon name="check" size={12} />直接证据</span>}
        </div>
      </div>
      <Icon name="chevron" />
    </button>
  );
}

function EvidenceDrawer({ videoId, hit, evidence, trace, onClose, onSeek }: { videoId: string; hit: SearchHit; evidence: EvidenceResponse | null; trace: AgentTrace; onClose: () => void; onSeek: () => void }) {
  const [tab, setTab] = useState<"evidence" | "trace" | "boundary">("evidence");
  const scene = hit.candidate;
  const traceSteps = [
    ["查询规划", `${trace.planner.planner} · ${trace.planner.status}`],
    ["候选召回", `${Number(trace.retrieval.candidate_count ?? trace.retrieval.recalled_count ?? 0)} 个候选`],
    ["证据验证", `${hit.verification.verifier} · ${hit.verification.verdict}`],
    ["最终排序", `Top ${trace.final_scene_ids.indexOf(scene.scene_id) + 1} · ${Math.round(hit.final_score * 100)}%`],
  ];
  return (
    <aside className="evidence-drawer">
      <div className="drawer-header">
        <div><span className="eyebrow">{scene.scene_id}</span><h2>{scene.summary}</h2></div>
        <button className="icon-button" onClick={onClose} aria-label="关闭"><Icon name="close" /></button>
      </div>
      <div className="drawer-actions">
        <button className="primary-button" onClick={onSeek}><Icon name="play" size={15} />跳到 {timecode(scene.start_ms)}</button>
        <span>{timecode(scene.start_ms)} — {timecode(scene.end_ms)}</span>
      </div>
      <div className="drawer-tabs">
        <button className={tab === "evidence" ? "active" : ""} onClick={() => setTab("evidence")}>证据</button>
        <button className={tab === "trace" ? "active" : ""} onClick={() => setTab("trace")}>Agent 轨迹</button>
        <button className={tab === "boundary" ? "active" : ""} onClick={() => setTab("boundary")}>边界</button>
      </div>
      <div className="drawer-content">
        {tab === "evidence" && <>
          <section className="evidence-block"><div className="block-title"><Icon name="evidence" /><span>视觉事件</span><em>{evidence?.visual ? `${Math.round(evidence.visual.confidence * 100)}%` : "--"}</em></div><p>{evidence?.visual?.summary ?? "正在读取视觉证据…"}</p><div className="token-row">{[...(evidence?.visual?.actions ?? []), ...(evidence?.visual?.objects ?? [])].map((item) => <span key={item}>{item}</span>)}</div></section>
          <section className="evidence-block"><div className="block-title"><span className="dialogue-icon">“</span><span>对白证据</span><em>{evidence?.dialogue.length ?? 0} 条</em></div>{evidence?.dialogue.length ? evidence.dialogue.map((item) => <blockquote key={item.utterance_id}><time>{timecode(item.start_ms)}</time>{item.text}</blockquote>) : <p className="muted">这个命中主要由画面证据支持，没有相邻对白。</p>}</section>
          <section className="confidence-grid">{Object.entries(hit.verification.components).filter(([key]) => ["visual_score", "dialogue_score", "entity_score", "temporal_score", "boundary_quality"].includes(key)).map(([key, value]) => <div key={key}><span>{key.replace("_score", "").replace("boundary_quality", "boundary")}</span><strong>{Math.round(value * 100)}</strong><i style={{ width: `${value * 100}%` }} /></div>)}</section>
        </>}
        {tab === "trace" && <div className="trace-list">{traceSteps.map(([title, detail], index) => <div className="trace-step" key={title}><span>{index + 1}</span><div><strong>{title}</strong><p>{detail}</p></div></div>)}<div className="trace-footer"><StatusPill value={trace.status} /><span>总耗时 {trace.total_latency_ms.toFixed(1)} ms</span></div></div>}
        {tab === "boundary" && <div className="boundary-panel"><div><span>模型原始区间</span><strong>{timecode(evidence?.boundary.raw_start_ms)}<small>→</small>{timecode(evidence?.boundary.raw_end_ms)}</strong></div><div className="boundary-arrow">SHOT-FIRST</div><div className="final-boundary"><span>镜头校准区间</span><strong>{timecode(evidence?.boundary.final_start_ms)}<small>→</small>{timecode(evidence?.boundary.final_end_ms)}</strong></div><dl><dt>入点修正</dt><dd>{evidence?.boundary.start_delta_frames ?? "--"} 帧</dd><dt>出点修正</dt><dd>{evidence?.boundary.end_delta_frames ?? "--"} 帧</dd><dt>策略</dt><dd>{evidence?.boundary.strategy ?? "--"}</dd></dl></div>}
      </div>
    </aside>
  );
}

export default function App() {
  const [videos, setVideos] = useState<VideoRecord[]>([]);
  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [selectedVideoId, setSelectedVideoId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [trace, setTrace] = useState<AgentTrace | null>(null);
  const [selectedHit, setSelectedHit] = useState<SearchHit | null>(null);
  const [evidence, setEvidence] = useState<EvidenceResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [searching, setSearching] = useState(false);
  const [searchAttempted, setSearchAttempted] = useState(false);
  const [diagnostics, setDiagnostics] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const refresh = useCallback(async () => {
    try {
      const workspace = await loadWorkspace();
      setVideos(workspace.videos);
      setJobs(workspace.jobs);
      setHealth(workspace.health);
      setSelectedVideoId((current) => current ?? workspace.videos[0]?.video_id ?? null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "无法连接 Runtime");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => { const timer = window.setInterval(() => void refresh(), 5000); return () => window.clearInterval(timer); }, [refresh]);

  const selectedVideo = useMemo(() => videos.find((item) => item.video_id === selectedVideoId) ?? null, [videos, selectedVideoId]);
  const activeJob = useMemo(() => jobs.find((item) => item.video_id === selectedVideoId && !terminalStates.includes(item.state)) ?? jobs.find((item) => item.video_id === selectedVideoId) ?? null, [jobs, selectedVideoId]);

  useEffect(() => {
    if (!activeJob || terminalStates.includes(activeJob.state)) return;
    return listenToJob(activeJob.job_id, (event: JobEvent) => {
      setJobs((current) => current.map((job) => job.job_id === event.job_id ? { ...job, state: event.state, progress: event.progress, completed_units: event.completed_units, total_units: event.total_units, message: event.message, updated_at: event.created_at } : job));
    }, () => void refresh());
  }, [activeJob?.job_id, activeJob?.state, refresh]);

  const handleUpload = async (file: File) => {
    setUploading(true); setError(null); setHits([]); setTrace(null); setSelectedHit(null); setSearchAttempted(false);
    try {
      const result = await uploadVideo(file);
      setSelectedVideoId(result.video.video_id);
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "上传失败");
    } finally { setUploading(false); }
  };

  const runSearch = async (event?: FormEvent, value = query) => {
    event?.preventDefault();
    const searchQuery = value.trim();
    if (!selectedVideo || !searchQuery) return;
    setSearching(true); setSearchAttempted(true); setError(null); setHits([]); setTrace(null); setSelectedHit(null); setEvidence(null);
    try {
      const result = await searchVideo(selectedVideo.video_id, searchQuery);
      setHits(result.hits); setTrace(result.trace);
      if (result.hits[0]) {
        await openHit(result.hits[0], result.trace);
        seek(result.hits[0]);
      }
    } catch (reason) {
      setSearchAttempted(false);
      setError(reason instanceof Error ? reason.message : "检索失败");
    } finally { setSearching(false); }
  };

  const openHit = async (hit: SearchHit, nextTrace = trace) => {
    if (!selectedVideo || !nextTrace) return;
    setSelectedHit(hit); setEvidence(null);
    try { setEvidence(await loadEvidence(selectedVideo.video_id, hit.candidate.scene_id)); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "证据读取失败"); }
  };

  const seek = (hit: SearchHit) => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = hit.candidate.start_ms / 1000;
    void videoRef.current.play().catch(() => undefined);
  };

  const chooseSuggestion = (value: string) => { setQuery(value); void runSearch(undefined, value); };
  const ready = selectedVideo?.status === "READY" || selectedVideo?.status === "PARTIAL";
  const noResults = noResultCopy(trace);
  const showingSearchResult = searching || searchAttempted || trace?.query === query.trim();

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand"><div className="brand-mark"><span /></div><div><strong>ShotSeek</strong><small>证据对齐场景检索</small></div></div>
        <div className="project-switcher">
          {videos.length > 0 ? <select value={selectedVideoId ?? ""} onChange={(event) => { setSelectedVideoId(event.target.value); setQuery(""); setHits([]); setTrace(null); setSelectedHit(null); setEvidence(null); setSearchAttempted(false); }} aria-label="选择视频">{videos.map((video) => <option key={video.video_id} value={video.video_id}>{video.original_filename}</option>)}</select> : <span>尚无视频</span>}
          {selectedVideo && <StatusPill value={selectedVideo.status} />}
        </div>
        <nav><select className="export-select" defaultValue="" disabled={!ready || !selectedVideo} aria-label="导出场景" onChange={(event) => { const format = event.currentTarget.value as "json" | "srt" | "xml" | "edl"; if (format && selectedVideo) window.location.assign(exportUrl(selectedVideo.video_id, format, hits.map((hit) => hit.candidate.scene_id))); event.currentTarget.value = ""; }}><option value="" disabled>{hits.length ? "导出命中 " + hits.length : "导出全部"}</option><option value="edl">CMX3600 EDL</option><option value="json">JSON</option><option value="srt">SRT</option><option value="xml">XML</option></select><label className="topbar-upload"><Icon name="upload" />添加视频<input type="file" accept="video/*,.mkv" onChange={(event) => { const file = event.target.files?.[0]; if (file) void handleUpload(file); }} /></label><button className="icon-button" onClick={() => setDiagnostics(true)} aria-label="诊断"><Icon name="settings" /></button></nav>
      </header>

      <main className={`workspace ${selectedHit ? "drawer-open" : ""}`}>
        {loading ? <div className="loading-screen"><div className="spinner" />正在连接 Production Runtime</div> : videos.length === 0 ? <div className="empty-workspace"><div className="hero-copy"><span className="eyebrow">EVIDENCE-ALIGNED TIMELINE RETRIEVAL</span><h1>一句话，找到<br /><em>准确的那一镜。</em></h1><p>ShotSeek 把自然语言问题映射到真实影视时间线，并给出画面、对白和镜头边界证据。</p></div><UploadSurface onFile={(file) => void handleUpload(file)} busy={uploading} /></div> : <div className="workbench">
          <section className="viewer-column">
            {activeJob && !terminalStates.includes(activeJob.state) && <ProcessingPanel job={activeJob} onCancel={() => void cancelJob(activeJob.job_id).then(refresh)} />}
            <div className="viewer-frame">
              {ready && selectedVideo ? <video key={selectedVideo.video_id} ref={videoRef} src={mediaUrl(selectedVideo.video_id)} controls preload="metadata" /> : <div className="viewer-wait"><span>{activeJob ? Math.round(activeJob.progress * 100) : 0}%</span><p>代理视频准备完成后将在这里播放</p></div>}
              <div className="viewer-overlay"><span>{selectedVideo?.width}×{selectedVideo?.height}</span><span>{selectedVideo?.fps?.toFixed(2)} FPS</span></div>
            </div>
            <div className="timeline-strip">
              <div className="timeline-labels"><span>00:00</span><strong>命中时间线</strong><span>{timecode(selectedVideo?.duration_ms).slice(0, 8)}</span></div>
              <div className="timeline-rail">{hits.map((hit, index) => <button key={hit.candidate.scene_id} className={selectedHit?.candidate.scene_id === hit.candidate.scene_id ? "active" : ""} style={{ left: `${Math.min(99, hit.candidate.start_ms / (selectedVideo?.duration_ms || 1) * 100)}%` }} onClick={() => { void openHit(hit); seek(hit); }} title={hit.candidate.summary}><span>{index + 1}</span></button>)}</div>
            </div>
            <div className="media-meta"><div><span>片长</span><strong>{timecode(selectedVideo?.duration_ms)}</strong></div><div><span>场景</span><strong>{selectedVideo?.scene_count ?? 0}</strong></div><div><span>文件</span><strong>{selectedVideo ? fileSize(selectedVideo.bytes) : "--"}</strong></div><div><span>索引</span><strong>{ready ? "FTS5 · READY" : "BUILDING"}</strong></div></div>
          </section>

          <section className="search-column">
            <div className="search-heading"><div><span className="eyebrow">SCENE FINDER</span><h1>寻找画面</h1></div>{trace && <StatusPill value={trace.status} />}</div>
            <form className="search-box" onSubmit={(event) => void runSearch(event)}><Icon name="search" size={20} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="描述对白、动作、人物或时间关系…" disabled={!ready} /><button className="search-submit" disabled={!ready || searching || !query.trim()}>{searching ? <span className="button-spinner" /> : "搜索"}</button></form>
            {!showingSearchResult ? (
              <div className="search-empty">
                <p>{ready ? "当前样片可试" : "视频处理完成后即可搜索"}</p>
                {ready && (
                  <div className="suggestions">
                    <button onClick={() => chooseSuggestion("Memory override in progress")}>对白：“Memory override in progress”</button>
                    <button onClick={() => chooseSuggestion("找到瞄准步枪的人")}>找到瞄准步枪的人</button>
                    <button onClick={() => chooseSuggestion("找到戴眼镜的人看全息屏幕")}>找到戴眼镜的人看全息屏幕</button>
                  </div>
                )}
                <div className="search-principle"><Icon name="trace" /><div><strong>答案必须有证据</strong><span>Planner → Retriever → Verifier → Shot-first</span></div></div>
              </div>
            ) : (
              <div className="results">
                <div className="results-meta"><span>{hits.length ? `找到 ${hits.length} 个可信场景` : searching ? "正在检索证据…" : noResults.heading}</span>{trace && <span>{trace.total_latency_ms.toFixed(0)} ms</span>}</div>
                {hits.map((hit, index) => <ResultCard key={hit.candidate.scene_id} hit={hit} index={index} videoId={selectedVideo!.video_id} active={selectedHit?.candidate.scene_id === hit.candidate.scene_id} onOpen={() => { void openHit(hit); seek(hit); }} />)}
                {!hits.length && !searching && <div className="no-results">{noResults.detail}</div>}
              </div>
            )}
          </section>
        </div>}
      </main>

      {selectedHit && trace && selectedVideo && <EvidenceDrawer videoId={selectedVideo.video_id} hit={selectedHit} evidence={evidence} trace={trace} onClose={() => setSelectedHit(null)} onSeek={() => seek(selectedHit)} />}
      {diagnostics && <div className="modal-backdrop" onMouseDown={() => setDiagnostics(false)}><section className="diagnostics-modal" onMouseDown={(event) => event.stopPropagation()}><div className="drawer-header"><div><span className="eyebrow">RUNTIME DIAGNOSTICS</span><h2>系统状态</h2></div><button className="icon-button" onClick={() => setDiagnostics(false)}><Icon name="close" /></button></div><div className="diagnostic-status"><StatusPill value={health?.status.toUpperCase() ?? "OFFLINE"} /><span>SQLite {health?.registry.integrity_check ?? "--"}</span></div><div className="diagnostic-grid"><div><span>视频</span><strong>{health?.registry.counts.video ?? 0}</strong></div><div><span>任务</span><strong>{health?.registry.counts.job ?? 0}</strong></div><div><span>事件</span><strong>{health?.registry.counts.job_event ?? 0}</strong></div><div><span>产物</span><strong>{health?.registry.counts.artifact ?? 0}</strong></div></div><p className="diagnostic-note">首屏只呈现创作所需信息；运行日志、缓存与硬件指标统一留在诊断层。</p></section></div>}
      {error && <div className="error-toast"><span>{error}</span><button onClick={() => setError(null)}><Icon name="close" size={15} /></button></div>}
    </div>
  );
}
