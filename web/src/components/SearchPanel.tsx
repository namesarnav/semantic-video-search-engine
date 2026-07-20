import { useState, type FormEvent } from "react";
import { Loader2, Search as SearchIcon } from "lucide-react";
import { searchVideos, type SearchResponse } from "@/lib/api";
import { groupByVideo } from "@/lib/group";
import { VideoResultCard } from "./VideoResultCard";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

/** Merging window used when "group near-duplicates" is on. A long static shot
 *  otherwise fills the list with the same moment several times over. */
const COLLAPSE_WINDOW_SEC = 3;

export function SearchPanel() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(25);
  const [collapse, setCollapse] = useState(true);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<SearchResponse | null>(null);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || searching) return;

    setSearching(true);
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

  const groups = response ? groupByVideo(response.results) : [];

  return (
    <div className="flex flex-col gap-5">
      <form onSubmit={onSubmit} role="search" className="flex flex-col gap-3">
        <div className="flex gap-2">
          <div className="relative flex-1">
            <SearchIcon
              className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-subtle"
              aria-hidden
            />
            <Input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="a red car driving at night"
              aria-label="Search query"
              className="pl-9"
            />
          </div>
          <Button type="submit" disabled={searching || !query.trim()}>
            {searching ? (
              <>
                <Loader2 className="size-4 animate-spin" aria-hidden />
                Searching…
              </>
            ) : (
              "Search"
            )}
          </Button>
        </div>

        <div className="flex flex-wrap items-center gap-4 text-xs text-muted">
          <label className="flex items-center gap-2" htmlFor="top-k">
            Results
            <select
              id="top-k"
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="rounded-md border border-line bg-surface-2 px-2 py-1 text-fg"
            >
              {[10, 25, 50].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <label className="flex cursor-pointer items-center gap-2">
            <input
              type="checkbox"
              checked={collapse}
              onChange={(e) => setCollapse(e.target.checked)}
              className="size-3.5 accent-accent"
            />
            Merge near-duplicates
          </label>
        </div>
      </form>

      {error && (
        <p
          role="alert"
          className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200"
        >
          {error}
        </p>
      )}

      {response && !error && (
        <p className="text-xs text-subtle nums">
          {groups.length} {groups.length === 1 ? "video" : "videos"} ·{" "}
          {response.count} {response.count === 1 ? "match" : "matches"} ·{" "}
          {response.took_ms} ms
        </p>
      )}

      {response && response.count === 0 && !error && (
        <p className="py-10 text-center text-sm text-subtle">
          No matches. Try describing what is visible in the frame.
        </p>
      )}

      <div className="flex flex-col gap-4">
        {groups.map((group) => (
          <VideoResultCard key={group.videoId} group={group} />
        ))}
      </div>
    </div>
  );
}
