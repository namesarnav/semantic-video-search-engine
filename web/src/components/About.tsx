import { Card, CardContent } from "@/components/ui/card";

const PIPELINE = [
  ["Sample", "Roughly one frame per second, plus extra frames at detected scene cuts, so short distinct moments are not skipped."],
  ["Embed", "CLIP encodes each frame into a vector. The same model encodes your query text into the same space — that shared space is the whole trick."],
  ["Index", "Vectors go to FAISS; every frame's video, timestamp and thumbnail go to SQLite, joined on the vector's position in the index."],
  ["Search", "Your query is embedded, matched by cosine similarity, and the winning vectors are joined back to the video and timestamp they came from."],
];

export function About() {
  return (
    <div className="flex flex-col gap-6">
      <Card>
        <CardContent className="pt-5">
          <h3 className="text-sm font-medium text-fg">
            Search video by what is in the picture
          </h3>
          <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted">
            Point it at a folder of videos, describe a moment in plain language,
            and get back ranked timestamps across every video you have ingested.
            It searches <strong className="text-fg">visual content</strong> — it
            is deliberately not transcript search, not OCR, and not caption
            search.
          </p>
        </CardContent>
      </Card>

      <div>
        <h3 className="mb-3 text-sm font-medium text-fg">How it works</h3>
        <ol className="grid gap-3 sm:grid-cols-2">
          {PIPELINE.map(([step, description], i) => (
            <li
              key={step}
              className="rounded-lg border border-line bg-surface-1 p-4"
            >
              <div className="flex items-center gap-2">
                <span className="text-xs text-subtle nums">0{i + 1}</span>
                <h4 className="text-sm font-medium text-fg">{step}</h4>
              </div>
              <p className="mt-1.5 text-xs leading-relaxed text-muted">
                {description}
              </p>
            </li>
          ))}
        </ol>
      </div>

      <div>
        <h3 className="mb-3 text-sm font-medium text-fg">Built with</h3>
        <div className="flex flex-wrap gap-2">
          {["CLIP ViT-B/32", "FAISS", "SQLite", "FastAPI", "React", "Docker"].map(
            (tech) => (
              <span
                key={tech}
                className="rounded-md border border-line bg-surface-1 px-2.5 py-1 text-xs text-muted"
              >
                {tech}
              </span>
            ),
          )}
        </div>
        <p className="mt-4 max-w-2xl text-xs leading-relaxed text-subtle">
          Quality is measured, not eyeballed: a hand-labelled eval set scores
          Recall@1/@5/@10, and design decisions like scene-aware sampling were
          settled by A/B against it rather than by assumption.
        </p>
      </div>
    </div>
  );
}
