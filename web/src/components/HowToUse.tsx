import { Card, CardContent } from "@/components/ui/card";

const STEPS = [
  {
    n: 1,
    title: "Ingest some video",
    body: "In the Library tab, give the server a path it can see, or upload a file. Sampling, embedding and indexing happen in the background — the status chip goes queued → processing → done.",
  },
  {
    n: 2,
    title: "Describe what is on screen",
    body: "Search for what is visible: “a person opens a laptop”, “red car at night”. Not what is said, not text in the frame — this searches pictures, not audio or captions.",
  },
  {
    n: 3,
    title: "Open the moment",
    body: "Each result is a video, named, with its matching moments beneath it. Click a thumbnail to jump the player straight to that timestamp.",
  },
];

const TIPS = [
  ["Describe, do not name", "“a snow covered ridge” beats “that ski clip” — the model matches appearance, not your filenames."],
  ["Merge near-duplicates", "On by default. A long static shot otherwise returns the same moment many times over."],
  ["Scissors icon", "That moment was sampled at a detected scene cut rather than on the once-a-second baseline."],
  ["Timestamps are approximate", "Sampling is roughly one frame per second, so a hit lands within about a second of the real moment."],
];

export function HowToUse() {
  return (
    <div className="flex flex-col gap-6">
      <div className="grid gap-3 sm:grid-cols-3">
        {STEPS.map((step) => (
          <Card key={step.n}>
            <CardContent className="pt-5">
              <span className="inline-flex size-7 items-center justify-center rounded-full bg-accent/15 text-xs font-semibold text-accent nums">
                {step.n}
              </span>
              <h3 className="mt-3 text-sm font-medium text-fg">{step.title}</h3>
              <p className="mt-1.5 text-xs leading-relaxed text-muted">
                {step.body}
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      <div>
        <h3 className="mb-3 text-sm font-medium text-fg">Getting better results</h3>
        <dl className="grid gap-3 sm:grid-cols-2">
          {TIPS.map(([term, description]) => (
            <div
              key={term}
              className="rounded-lg border border-line bg-surface-1 p-4"
            >
              <dt className="text-sm font-medium text-fg">{term}</dt>
              <dd className="mt-1 text-xs leading-relaxed text-muted">
                {description}
              </dd>
            </div>
          ))}
        </dl>
      </div>

      <div className="rounded-lg border border-line bg-surface-1 p-4">
        <h3 className="text-sm font-medium text-fg">From the command line</h3>
        <pre className="mt-3 overflow-x-auto rounded-md bg-bg p-3 font-mono text-xs leading-relaxed text-muted">
{`sv-engine index data/videos          # ingest a file or folder
sv-engine search "a red car" -k 10   # query
sv-engine videos                     # status per video
sv-engine eval                       # Recall@K against the eval set`}
        </pre>
      </div>
    </div>
  );
}
