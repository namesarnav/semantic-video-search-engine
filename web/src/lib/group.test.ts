import { describe, expect, it } from "vitest";
import { groupByVideo } from "./group";
import type { SearchHit } from "./api";

function hit(partial: Partial<SearchHit> & { video_id: string }): SearchHit {
  return {
    score: 0.5,
    filename: `${partial.video_id}.mp4`,
    timestamp_sec: 0,
    thumbnail_url: "/thumbnails/1",
    video_url: `/videos/${partial.video_id}/file`,
    reason: "baseline",
    frame_id: 1,
    ...partial,
  };
}

describe("groupByVideo", () => {
  it("returns one group per video, not one per frame", () => {
    const groups = groupByVideo([
      hit({ video_id: "a", frame_id: 1 }),
      hit({ video_id: "a", frame_id: 2 }),
      hit({ video_id: "b", frame_id: 3 }),
    ]);

    expect(groups).toHaveLength(2);
    expect(groups.map((g) => g.videoId)).toEqual(["a", "b"]);
  });

  it("carries the filename and video url onto the group", () => {
    const [group] = groupByVideo([
      hit({ video_id: "a", filename: "beach.mp4", video_url: "/videos/a/file" }),
    ]);

    expect(group.filename).toBe("beach.mp4");
    expect(group.videoUrl).toBe("/videos/a/file");
  });

  it("orders groups by their best-scoring moment, not by first appearance", () => {
    const groups = groupByVideo([
      hit({ video_id: "a", score: 0.4 }),
      hit({ video_id: "b", score: 0.9 }),
      hit({ video_id: "a", score: 0.5 }),
    ]);

    expect(groups.map((g) => g.videoId)).toEqual(["b", "a"]);
    expect(groups[0].bestScore).toBe(0.9);
  });

  it("orders moments within a group by timestamp, so they read as a timeline", () => {
    const [group] = groupByVideo([
      hit({ video_id: "a", timestamp_sec: 9, score: 0.9 }),
      hit({ video_id: "a", timestamp_sec: 2, score: 0.3 }),
      hit({ video_id: "a", timestamp_sec: 5, score: 0.6 }),
    ]);

    expect(group.moments.map((m) => m.timestamp_sec)).toEqual([2, 5, 9]);
  });

  it("keeps the best-scoring moment reachable for the poster frame", () => {
    const [group] = groupByVideo([
      hit({ video_id: "a", timestamp_sec: 1, score: 0.2, frame_id: 10 }),
      hit({ video_id: "a", timestamp_sec: 8, score: 0.8, frame_id: 20 }),
    ]);

    expect(group.bestMoment.frame_id).toBe(20);
    expect(group.bestScore).toBe(0.8);
  });

  it("counts the moments in the group", () => {
    const [group] = groupByVideo([
      hit({ video_id: "a", frame_id: 1 }),
      hit({ video_id: "a", frame_id: 2 }),
    ]);

    expect(group.moments).toHaveLength(2);
  });

  it("handles an empty result set", () => {
    expect(groupByVideo([])).toEqual([]);
  });
});
