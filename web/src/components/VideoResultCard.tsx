import { useRef, useState } from "react";
import { Scissors, Film } from "lucide-react";
import type { VideoGroup } from "@/lib/group";
import { formatTimestamp } from "@/lib/format";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

/**
 * One matched video: the file itself, named, with its matching moments laid
 * along it.
 *
 * The engine ranks frames, but a frame is evidence, not the answer -- the
 * answer is "this video, at these times". So the video is the unit here, and
 * the thumbnails are demoted to a strip of seek targets under the player.
 * Clicking one scrubs rather than navigating, which keeps a result explorable
 * without leaving the page.
 */
export function VideoResultCard({ group }: { group: VideoGroup }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [active, setActive] = useState(group.bestMoment.frame_id);

  function seekTo(timestampSec: number, frameId: number) {
    setActive(frameId);
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = timestampSec;
    // A click is an explicit request to see the moment, so start playing --
    // but a rejected promise here (autoplay policy, no codec) must not become
    // an unhandled rejection in the console.
    void video.play?.()?.catch(() => {});
  }

  return (
    <Card className="overflow-hidden transition-colors hover:border-line/80">
      <div className="flex flex-col gap-4 p-4 sm:flex-row">
        <div className="relative w-full shrink-0 overflow-hidden rounded-lg bg-black sm:w-[22rem]">
          <video
            ref={videoRef}
            src={group.videoUrl}
            poster={group.bestMoment.thumbnail_url}
            controls
            preload="metadata"
            className="aspect-video h-full w-full"
          />
        </div>

        <div className="flex min-w-0 flex-1 flex-col gap-3">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <h3
                className="flex items-center gap-2 truncate font-medium text-fg"
                title={group.filename}
              >
                <Film className="size-4 shrink-0 text-subtle" aria-hidden />
                {group.filename}
              </h3>
              <p className="mt-1 text-xs text-subtle nums">
                {group.moments.length}{" "}
                {group.moments.length === 1 ? "moment" : "moments"} matched ·
                best at {formatTimestamp(group.bestMoment.timestamp_sec)}
              </p>
            </div>
            <Badge tone="accent" className="shrink-0 nums">
              {group.bestScore.toFixed(3)}
            </Badge>
          </div>

          <div className="flex flex-wrap gap-2">
            {group.moments.map((moment) => (
              <button
                key={moment.frame_id}
                type="button"
                onClick={() => seekTo(moment.timestamp_sec, moment.frame_id)}
                className={cn(
                  "group relative overflow-hidden rounded-md border transition-all",
                  "focus-visible:outline-none focus-visible:ring-2",
                  "focus-visible:ring-accent/60",
                  active === moment.frame_id
                    ? "border-accent ring-1 ring-accent/40"
                    : "border-line hover:border-subtle",
                )}
              >
                <img
                  src={moment.thumbnail_url}
                  alt=""
                  loading="lazy"
                  className="h-14 w-24 object-cover"
                />
                <span
                  className={cn(
                    "absolute inset-x-0 bottom-0 flex items-center justify-between",
                    "gap-1 bg-black/70 px-1.5 py-0.5 text-[10px] text-white nums",
                  )}
                >
                  {formatTimestamp(moment.timestamp_sec)}
                  {moment.reason === "scene_cut" && (
                    // Titled rather than labelled: it annotates the moment,
                    // while the timestamp is what names the button.
                    <span title="scene cut" className="flex items-center">
                      <Scissors className="size-2.5 text-accent" aria-hidden />
                    </span>
                  )}
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </Card>
  );
}
