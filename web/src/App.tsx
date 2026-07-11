import { useState, type FormEvent } from "react";
import { searchVideos, type SearchResponse } from "./api";
import { formatTimestamp } from "./format";

/** Seconds within which hits from one video are treated as the same moment.
 * Matches the CLI's `--collapse` default; a long static shot otherwise floods
 * the grid with near-identical frames. */
const COLLAPSE_WINDOW_SEC = 3;

export default function App() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(10);
  const [collapse, setCollapse] = useState(false);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<SearchResponse | null>(null);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || searching) return;

    setSearching(true);
    // Drop the previous outcome now. Leaving stale hits on screen under a new
    // query reads as though they answer it.
    setError(null);
    setResponse(null);
    try {
      setResponse(
        await searchVideos(trimmed, {
          topK,
          collapseWindowSec: collapse ? COLLAPSE_WINDOW_SEC : undefined,
        }),
      );
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setSearching(false);
    }
  }

  return (
    <main className="page">
      <header className="masthead">
        <h1>Semantic video search</h1>
        <p className="tagline">
          Describe what happens on screen — “a person opens a laptop”, “red car at night”.
        </p>
      </header>

      <form className="controls" onSubmit={onSubmit} role="search">
        <input
          type="search"
          className="query"
          placeholder="Describe a moment…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          autoFocus
        />
        <button type="submit" className="go" disabled={searching}>
          {searching ? "Searching…" : "Search"}
        </button>

        <div className="options">
          <label htmlFor="top-k">Results</label>
          <select
            id="top-k"
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
          >
            {[10, 25, 50].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>

          <label className="check">
            <input
              type="checkbox"
              checked={collapse}
              onChange={(e) => setCollapse(e.target.checked)}
            />
            Merge near-duplicates
          </label>
        </div>
      </form>

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      {response && !error && (
        <>
          <p className="meta">
            {`${response.count} ${response.count === 1 ? "match" : "matches"} · ${response.took_ms} ms`}
          </p>
          {response.count === 0 ? (
            <p className="empty">
              No matches. Try describing the picture rather than naming it — the index
              is built from what a frame looks like, not from speech or on-screen text.
            </p>
          ) : (
            <ul className="grid">
              {response.results.map((hit) => (
                <li key={hit.frame_id} className="hit">
                  <img
                    src={hit.thumbnail_url}
                    alt={`${hit.filename} at ${formatTimestamp(hit.timestamp_sec)}`}
                    loading="lazy"
                  />
                  <div className="caption">
                    <span className="time">{formatTimestamp(hit.timestamp_sec)}</span>
                    <span className="score">{hit.score.toFixed(3)}</span>
                  </div>
                  <div className="source">
                    <span className="filename">{hit.filename}</span>
                    <span className={`reason ${hit.reason}`}>{hit.reason}</span>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </main>
  );
}
