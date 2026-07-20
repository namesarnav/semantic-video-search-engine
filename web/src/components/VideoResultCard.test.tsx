import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { VideoResultCard } from "./VideoResultCard";
import { groupByVideo } from "@/lib/group";
import type { SearchHit } from "@/lib/api";

function hit(partial: Partial<SearchHit> & { video_id: string }): SearchHit {
  return {
    score: 0.5,
    filename: "beach_sunset.mp4",
    timestamp_sec: 0,
    reason: "baseline",
    frame_id: 1,
    ...partial,
    // Derived after the spread so it always tracks frame_id, rather than
    // pinning a value the assertions would then be testing instead of code.
    thumbnail_url: `/thumbnails/${partial.frame_id ?? 1}`,
    video_url: `/videos/${partial.video_id}/file`,
  };
}

const group = groupByVideo([
  hit({ video_id: "a", timestamp_sec: 5, score: 0.9, frame_id: 11 }),
  hit({ video_id: "a", timestamp_sec: 72, score: 0.4, frame_id: 12 }),
])[0];

describe("VideoResultCard", () => {
  it("shows the video's filename", () => {
    render(<VideoResultCard group={group} />);
    expect(screen.getByText("beach_sunset.mp4")).toBeInTheDocument();
  });

  it("renders the video itself, not just frames", () => {
    const { container } = render(<VideoResultCard group={group} />);
    const video = container.querySelector("video");

    expect(video).not.toBeNull();
    expect(video).toHaveAttribute("src", "/videos/a/file");
  });

  it("starts the player at the best-matching moment", () => {
    const { container } = render(<VideoResultCard group={group} />);
    // 0.9 at 5s beats 0.4 at 72s, so the player opens on 5s rather than 0.
    expect(container.querySelector("video")).toHaveAttribute(
      "poster",
      "/thumbnails/11",
    );
  });

  it("lists every matched moment as a readable timestamp", () => {
    render(<VideoResultCard group={group} />);
    expect(screen.getByRole("button", { name: /0:05/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /1:12/ })).toBeInTheDocument();
  });

  it("seeks the player when a moment is clicked", async () => {
    const { container } = render(<VideoResultCard group={group} />);
    const video = container.querySelector("video") as HTMLVideoElement;
    // jsdom does not implement playback; stub what the handler touches.
    video.play = vi.fn().mockResolvedValue(undefined);

    await userEvent.click(screen.getByRole("button", { name: /1:12/ }));

    expect(video.currentTime).toBe(72);
  });

  it("says how many moments matched", () => {
    render(<VideoResultCard group={group} />);
    expect(screen.getByText(/2 moments/i)).toBeInTheDocument();
  });

  it("uses the singular for one moment", () => {
    const single = groupByVideo([hit({ video_id: "b", timestamp_sec: 3 })])[0];
    render(<VideoResultCard group={single} />);
    expect(screen.getByText(/1 moment(?!s)/i)).toBeInTheDocument();
  });

  it("marks the moment that came from a scene cut", () => {
    const cuts = groupByVideo([
      hit({ video_id: "c", timestamp_sec: 4, reason: "scene_cut" }),
    ])[0];
    render(<VideoResultCard group={cuts} />);

    const moment = screen.getByRole("button", { name: /0:04/ });
    expect(within(moment).getByTitle(/scene cut/i)).toBeInTheDocument();
  });
});
