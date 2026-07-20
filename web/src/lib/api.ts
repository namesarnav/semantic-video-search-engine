/** The one place that knows the shape of the backend.
 *
 * Paths are relative on purpose. The API returns `thumbnail_url` as a relative
 * path, the dev server proxies these prefixes, and in production FastAPI serves
 * the built files itself -- so the client never needs to be told where the
 * backend is, and there is no base URL to configure or get wrong.
 */

export interface SearchHit {
  score: number;
  video_id: string;
  filename: string;
  timestamp_sec: number;
  thumbnail_url: string;
  /** Source video, so a match can be played rather than only shown. */
  video_url: string;
  reason: string;
  frame_id: number;
}

export interface SearchResponse {
  query: string;
  took_ms: number;
  count: number;
  results: SearchHit[];
}

export interface SearchOptions {
  topK: number;
  collapseWindowSec?: number;
}

/** FastAPI puts the message in `detail`, as a string for HTTPException and a
 * list of field errors for a 422. Neither is worth showing raw. */
export interface VideoSummary {
  id: string;
  filename: string;
  status: "queued" | "processing" | "done" | "failed";
  duration_sec: number;
  frame_count: number;
  video_url: string;
  error: string | null;
  ingested_at: string | null;
}

async function messageFor(response: Response): Promise<string> {
  try {
    const body = await response.json();
    const detail = body?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail) && detail[0]?.msg) return String(detail[0].msg);
  } catch {
    // A proxy or a crashed worker can answer with HTML; fall through.
  }
  return `request failed (${response.status})`;
}

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) throw new Error(await messageFor(response));
  return (await response.json()) as T;
}

/** Everything the library tab shows: what is ingested and how it is doing. */
export async function listVideos(): Promise<VideoSummary[]> {
  const response = await fetch("/videos");
  const body = await json<{ videos: VideoSummary[] }>(response);
  return body.videos;
}

/** Queue a server-side path. Returns 202 -- the work happens afterwards, so
 * the caller has to poll rather than assume the video is ready. */
export async function ingestPath(
  path: string,
): Promise<{ video_id: string; filename: string; status: string }> {
  return json(
    await fetch("/videos", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: path.trim() }),
    }),
  );
}

export async function uploadVideo(
  file: File,
): Promise<{ video_id: string; filename: string; status: string }> {
  const form = new FormData();
  form.append("file", file);
  return json(await fetch("/videos/upload", { method: "POST", body: form }));
}

export interface Health {
  status: string;
  videos: number;
  frames: number;
  vectors: number;
  device: string;
}

export async function getHealth(): Promise<Health> {
  return json(await fetch("/health"));
}

export async function searchVideos(
  query: string,
  { topK, collapseWindowSec }: SearchOptions,
): Promise<SearchResponse> {
  const response = await fetch("/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      top_k: topK,
      // Omitted rather than null: the server treats absent as "no collapsing",
      // and sending null would have to be special-cased at both ends.
      ...(collapseWindowSec ? { collapse_window_sec: collapseWindowSec } : {}),
    }),
  });

  if (!response.ok) throw new Error(await messageFor(response));
  return (await response.json()) as SearchResponse;
}
