import type { SearchHit } from "./api";

/**
 * A video, and the moments inside it that matched.
 *
 * The engine ranks *frames*, because that is what is embedded and indexed.
 * But a person searching does not want twelve stills; they want the video the
 * moment is in, named, with the matches laid out along it. Grouping is
 * therefore a presentation concern and lives here rather than in the API --
 * `/search` keeps returning a flat ranked list, which is the honest shape of
 * what the index actually computed.
 */
export interface VideoGroup {
  videoId: string;
  filename: string;
  videoUrl: string;
  /** Matched moments, in timestamp order so they read as a timeline. */
  moments: SearchHit[];
  /** Highest-scoring moment: what the group is ranked by and postered with. */
  bestMoment: SearchHit;
  bestScore: number;
}

export function groupByVideo(hits: SearchHit[]): VideoGroup[] {
  const byVideo = new Map<string, SearchHit[]>();
  for (const hit of hits) {
    const existing = byVideo.get(hit.video_id);
    if (existing) existing.push(hit);
    else byVideo.set(hit.video_id, [hit]);
  }

  const groups: VideoGroup[] = [];
  for (const [videoId, moments] of byVideo) {
    // Ranking a video by its best moment, not by how many moments it has:
    // one strong match is a better answer than five weak ones, and counting
    // would just reward long videos.
    const bestMoment = moments.reduce((best, m) =>
      m.score > best.score ? m : best,
    );
    groups.push({
      videoId,
      filename: bestMoment.filename,
      videoUrl: bestMoment.video_url,
      moments: [...moments].sort((a, b) => a.timestamp_sec - b.timestamp_sec),
      bestMoment,
      bestScore: bestMoment.score,
    });
  }

  return groups.sort((a, b) => b.bestScore - a.bestScore);
}
