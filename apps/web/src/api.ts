import type {
  EvidenceResponse,
  HealthResponse,
  JobEvent,
  JobRecord,
  SearchResponse,
  VideoRecord,
} from "./types";

async function request<T>(input: RequestInfo | URL, init?: RequestInit): Promise<T> {
  const response = await fetch(input, init);
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    const detail = payload?.detail ?? `${response.status} ${response.statusText}`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return response.json() as Promise<T>;
}

export async function loadWorkspace(): Promise<{
  videos: VideoRecord[];
  jobs: JobRecord[];
  health: HealthResponse;
}> {
  const [videos, jobs, health] = await Promise.all([
    request<{ items: VideoRecord[] }>("/api/v1/videos"),
    request<{ items: JobRecord[] }>("/api/v1/jobs"),
    request<HealthResponse>("/health"),
  ]);
  return { videos: videos.items, jobs: jobs.items, health };
}

export async function uploadVideo(file: File): Promise<{
  video: VideoRecord;
  job: JobRecord;
  upload_created: boolean;
  job_reused: boolean;
}> {
  return request(`/api/v1/jobs?filename=${encodeURIComponent(file.name)}`, {
    method: "POST",
    headers: { "Content-Type": file.type || "application/octet-stream" },
    body: file,
  });
}

export async function cancelJob(jobId: string): Promise<JobRecord> {
  const payload = await request<{ job: JobRecord }>(
    `/api/v1/jobs/${jobId}/cancel`,
    { method: "POST" },
  );
  return payload.job;
}

export async function searchVideo(
  videoId: string,
  query: string,
): Promise<SearchResponse> {
  return request(`/api/v1/videos/${videoId}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      top_k: 3,
      planner_mode: "auto",
      verifier_mode: "auto",
    }),
  });
}

export function loadEvidence(
  videoId: string,
  sceneId: string,
): Promise<EvidenceResponse> {
  return request(`/api/v1/videos/${videoId}/scenes/${sceneId}/evidence`);
}

export function mediaUrl(videoId: string): string {
  return `/api/v1/videos/${videoId}/media?kind=proxy`;
}

export function previewUrl(videoId: string, sceneId: string): string {
  return `/api/v1/videos/${videoId}/scenes/${sceneId}/preview`;
}

export function listenToJob(
  jobId: string,
  onEvent: (event: JobEvent) => void,
  onEnd: () => void,
): () => void {
  const source = new EventSource(`/api/v1/jobs/${jobId}/events`);
  source.addEventListener("job", (event) => {
    onEvent(JSON.parse((event as MessageEvent).data) as JobEvent);
  });
  source.addEventListener("end", () => {
    source.close();
    onEnd();
  });
  source.onerror = () => source.close();
  return () => source.close();
}
