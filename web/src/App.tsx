import { useEffect, useState } from "react";
import { BookOpen, Info, Library, Search, Video } from "lucide-react";
import { getHealth, type Health } from "@/lib/api";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { SearchPanel } from "@/components/SearchPanel";
import { LibraryPanel } from "@/components/LibraryPanel";
import { HowToUse } from "@/components/HowToUse";
import { About } from "@/components/About";

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    // Best-effort: the corpus counters are context, not function. A failure
    // here must not take the page down or shout at the user, because
    // everything else still works.
    getHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  return (
    <div className="min-h-screen">
      <header className="border-b border-line/70 bg-surface-1/40 backdrop-blur">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center gap-3 px-6 py-4">
          <div className="flex items-center gap-2.5">
            <span className="flex size-8 items-center justify-center rounded-lg bg-accent/15">
              <Video className="size-4 text-accent" aria-hidden />
            </span>
            <div>
              <h1 className="text-sm font-semibold leading-tight text-fg">
                Semantic Video Search
              </h1>
              <p className="text-xs leading-tight text-subtle">
                Find the moment, not the file name
              </p>
            </div>
          </div>

          {health && (
            <p className="ml-auto text-xs text-subtle nums">
              {health.videos} videos · {health.frames} frames ·{" "}
              <span className="text-muted">{health.device}</span>
            </p>
          )}
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-6">
        <Tabs defaultValue="search" className="flex flex-col gap-6">
          <TabsList>
            <TabsTrigger value="search">
              <Search className="size-3.5" aria-hidden />
              Search
            </TabsTrigger>
            <TabsTrigger value="library">
              <Library className="size-3.5" aria-hidden />
              Library
            </TabsTrigger>
            <TabsTrigger value="how">
              <BookOpen className="size-3.5" aria-hidden />
              How to use
            </TabsTrigger>
            <TabsTrigger value="about">
              <Info className="size-3.5" aria-hidden />
              About
            </TabsTrigger>
          </TabsList>

          <TabsContent value="search">
            <SearchPanel />
          </TabsContent>
          <TabsContent value="library">
            <LibraryPanel />
          </TabsContent>
          <TabsContent value="how">
            <HowToUse />
          </TabsContent>
          <TabsContent value="about">
            <About />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
