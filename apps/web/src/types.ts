export type JobState =
  | "CREATED"
  | "QUEUED"
  | "PROBING"
  | "TRANSCODING"
  | "EXTRACTING_AUDIO"
  | "DETECTING_SHOTS"
  | "CHUNKING"
  | "ANALYZING_VISUAL"
  | "ANALYZING_ASR"
  | "ALIGNING"
  | "BUILDING_SCENES"
  | "INDEXING"
  | "RETRYING"
  | "READY"
  | "PARTIAL"
  | "FAILED"
  | "CANCELLED";

export interface VideoRecord {
  video_id: string;
  sha256: string;
  original_filename: string;
  source_path: string;
  proxy_path: string | null;
  audio_path: string | null;
  duration_ms: number | null;
  width: number | null;
  height: number | null;
  fps: number | null;
  bytes: number;
  scene_count: number;
  search_db_path: string | null;
  status: "REGISTERED" | "PROCESSING" | "READY" | "PARTIAL" | "FAILED";
  created_at: string;
  updated_at: string;
}

export interface JobRecord {
  job_id: string;
  video_id: string;
  state: JobState;
  progress: number;
  current_stage: string;
  completed_units: number;
  total_units: number;
  retry_count: number;
  cancel_requested: boolean;
  error_code: string | null;
  message: string;
  resume_state: JobState | null;
  created_at: string;
  updated_at: string;
}

export interface JobEvent {
  event_id: number;
  job_id: string;
  state: JobState;
  progress: number;
  completed_units: number;
  total_units: number;
  message: string;
  created_at: string;
}

export interface CandidateScene {
  scene_id: string;
  start_ms: number;
  end_ms: number;
  start_frame: number;
  end_frame: number;
  summary: string;
  characters: string[];
  actions: string[];
  objects: string[];
  location: string | null;
  visible_text: string[];
  dialogue: string;
  shot_ids: string[];
  evidence_refs: Array<{ kind: string; evidence_id: string }>;
  retrieval_route: string;
  retrieval_score: number;
  components: Record<string, number>;
}

export interface SearchHit {
  candidate: CandidateScene;
  verification: {
    scene_id: string;
    verdict: "supported" | "unsupported" | "uncertain";
    direct_evidence: boolean;
    matched_constraints: string[];
    failed_constraints: string[];
    contradictions: string[];
    confidence: number;
    reason: string;
    verifier: "rule" | "stepfun" | "cache";
    components: Record<string, number>;
  };
  final_score: number;
}

export interface AgentTrace {
  trace_id: string;
  status: "LIVE" | "CACHED" | "FALLBACK" | "RULE";
  query: string;
  query_spec: {
    quoted_text: string | null;
    entities: Array<{ text: string; role: string }>;
    actions: string[];
    objects: string[];
    locations: string[];
    keywords: string[];
    temporal_constraints: unknown[];
    ordinal: { value: number | "last" } | null;
    evidence_preference: string[];
  };
  planner: {
    status: string;
    planner: string;
    route_reason: string;
    latency_ms: number;
    fallback_reason: string | null;
  };
  retrieval: Record<string, unknown>;
  temporal: Record<string, unknown>;
  verification: Record<string, unknown>;
  final_scene_ids: string[];
  phase_latency_ms: Record<string, number>;
  total_latency_ms: number;
}

export interface SearchResponse {
  hits: SearchHit[];
  trace: AgentTrace;
}

export interface EvidenceResponse {
  schema_version: string;
  scene_id: string;
  visual: null | {
    summary: string;
    confidence: number;
    source: string;
    model: string;
    actions: string[];
    objects: string[];
    characters: string[];
    location: string | null;
    visible_text: string[];
  };
  dialogue: Array<{
    utterance_id: string;
    start_ms: number;
    end_ms: number;
    text: string;
    speaker_id: string | null;
    source: string;
  }>;
  boundary: {
    strategy: string | null;
    raw_start_ms: number | null;
    raw_end_ms: number | null;
    final_start_ms: number;
    final_end_ms: number;
    start_delta_frames: number | null;
    end_delta_frames: number | null;
  };
  evidence_refs: Array<{ kind: string; evidence_id: string }>;
}

export interface HealthResponse {
  status: string;
  service: string;
  worker_enabled: boolean;
  registry: {
    integrity_check: string;
    counts: Record<string, number>;
  };
}

export const terminalStates: JobState[] = [
  "READY",
  "PARTIAL",
  "FAILED",
  "CANCELLED",
];
