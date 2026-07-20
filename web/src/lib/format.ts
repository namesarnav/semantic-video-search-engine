/** Seconds as `m:ss`, or `h:mm:ss` once a video runs past the hour.
 *
 * Floors rather than rounds: a hit points at a frame we actually sampled, and
 * rounding 59.9s up to 1:00 would send the viewer a second past it.
 */
export function formatTimestamp(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const pad = (n: number) => String(n).padStart(2, "0");
  const secs = total % 60;
  const mins = Math.floor(total / 60) % 60;
  const hours = Math.floor(total / 3600);
  return hours > 0 ? `${hours}:${pad(mins)}:${pad(secs)}` : `${mins}:${pad(secs)}`;
}
