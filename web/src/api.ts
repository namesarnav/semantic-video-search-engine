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
async function messageFor(response: Response): Promise<string> {
  try {
    const body = await response.json();
    const detail = body?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail) && detail[0]?.msg) return String(detail[0].msg);
  } catch {
    // A proxy or a crashed worker can answer with HTML; fall through.
  }
  return `search failed (${response.status})`;
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
