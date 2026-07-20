import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2, RefreshCw, Upload, Plus } from "lucide-react";
import { ingestPath, listVideos, uploadVideo, type VideoSummary } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

const TONE = {
  done: "good",
  failed: "bad",
  processing: "warn",
  queued: "neutral",
} as const;

/** Ingestion is a 202: the server takes the job and finishes it later, so the
 *  UI has to poll rather than assume. Only while something is in flight. */
const POLL_MS = 2000;

export function LibraryPanel() {
  const [videos, setVideos] = useState<VideoSummary[]>([]);
  const [path, setPath] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      setVideos(await listVideos());
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Poll only while a video is actually mid-ingest; a corpus at rest should
  // not generate traffic forever.
  const working = videos.some(
    (v) => v.status === "queued" || v.status === "processing",
  );
  useEffect(() => {
    if (!working) return;
    const timer = setInterval(() => void refresh(), POLL_MS);
    return () => clearInterval(timer);
  }, [working, refresh]);

  async function run(action: () => Promise<{ filename: string }>) {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const result = await action();
      setNotice(`Queued ${result.filename}. Ingesting in the background…`);
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-5">
      <Card>
        <CardContent className="flex flex-col gap-4 pt-5">
          <div>
            <h3 className="text-sm font-medium text-fg">Ingest a video</h3>
            <p className="mt-1 text-xs text-subtle">
              A path the <em>server</em> can see. Re-ingesting a video already
              indexed is a no-op — videos are keyed by content hash, not name.
            </p>
          </div>

          <div className="flex flex-col gap-2 sm:flex-row">
            <Input
              value={path}
              onChange={(e) => setPath(e.target.value)}
              placeholder="data/videos/clip.mp4"
              aria-label="Video path on the server"
              onKeyDown={(e) => {
                if (e.key === "Enter" && path.trim() && !busy)
                  void run(() => ingestPath(path));
              }}
            />
            <div className="flex gap-2">
              <Button
                onClick={() => void run(() => ingestPath(path))}
                disabled={busy || !path.trim()}
              >
                {busy ? (
                  <Loader2 className="size-4 animate-spin" aria-hidden />
                ) : (
                  <Plus className="size-4" aria-hidden />
                )}
                Ingest
              </Button>
              <Button
                variant="secondary"
                onClick={() => fileInput.current?.click()}
                disabled={busy}
              >
                <Upload className="size-4" aria-hidden />
                Upload
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => void refresh()}
                aria-label="Refresh video list"
                title="Refresh"
              >
                <RefreshCw className="size-4" aria-hidden />
              </Button>
            </div>
          </div>

          <input
            ref={fileInput}
            type="file"
            accept="video/*"
            className="hidden"
            aria-label="Upload a video file"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void run(() => uploadVideo(file));
              e.target.value = "";
            }}
          />

          {notice && <p className="text-xs text-accent">{notice}</p>}
          {error && (
            <p role="alert" className="text-xs text-rose-300">
              {error}
            </p>
          )}
        </CardContent>
      </Card>

      <div className="flex flex-col gap-2">
        <div className="flex items-baseline justify-between">
          <h3 className="text-sm font-medium text-fg">
            Library{" "}
            <span className="text-subtle nums">({videos.length})</span>
          </h3>
          {working && (
            <span className="flex items-center gap-1.5 text-xs text-amber-300">
              <Loader2 className="size-3 animate-spin" aria-hidden />
              ingesting…
            </span>
          )}
        </div>

        {videos.length === 0 ? (
          <p className="rounded-lg border border-dashed border-line py-10 text-center text-sm text-subtle">
            Nothing ingested yet.
          </p>
        ) : (
          <ul className="flex flex-col gap-2">
            {videos.map((video) => (
              <li
                key={video.id}
                className="flex items-center justify-between gap-3 rounded-lg border border-line bg-surface-1 px-4 py-3"
              >
                <div className="min-w-0">
                  <p className="truncate text-sm text-fg">{video.filename}</p>
                  <p className="text-xs text-subtle nums">
                    {video.frame_count} frames · {video.duration_sec.toFixed(1)}s
                    {video.error && (
                      <span className="text-rose-300"> · {video.error}</span>
                    )}
                  </p>
                </div>
                <Badge tone={TONE[video.status]}>{video.status}</Badge>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
