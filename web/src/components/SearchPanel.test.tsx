/**
 * What the search panel must do.
 *
 * The backend is stubbed at `fetch`: these test the UI's behaviour, not the
 * API, which has its own suite. What matters is that a query reaches the right
 * endpoint in the right shape, that results are presented as *videos* rather
 * than loose frames, and that no state leaves the page silently blank.
 */
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SearchPanel } from "./SearchPanel";
import type { SearchResponse } from "@/lib/api";

const hit = (over: Partial<SearchResponse["results"][0]> = {}) => ({
  score: 0.31,
  video_id: "abc123",
  filename: "kitchen.mp4",
  timestamp_sec: 83.5,
  thumbnail_url: "/thumbnails/42",
  video_url: "/videos/abc123/file",
  reason: "scene_cut",
  frame_id: 42,
  ...over,
});

const response = (over: Partial<SearchResponse> = {}): SearchResponse => ({
  query: "a person opens a laptop",
  took_ms: 12.5,
  count: 1,
  results: [hit()],
  ...over,
});

const ok = (body: unknown) =>
  Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) } as Response);

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  fetchMock = vi.fn(() => ok(response()));
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

const search = async (query: string) => {
  const user = userEvent.setup();
  await user.type(screen.getByRole("searchbox"), query);
  await user.click(screen.getByRole("button", { name: /^search$/i }));
  return user;
};

describe("searching", () => {
  it("posts the query to /search and renders the result", async () => {
    render(<SearchPanel />);
    await search("a person opens a laptop");

    await waitFor(() => expect(screen.getByText("kitchen.mp4")).toBeInTheDocument());

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/search");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toMatchObject({ query: "a person opens a laptop" });
  });

  it("presents a result as the video file, named, not as a loose frame", async () => {
    const { container } = render(<SearchPanel />);
    await search("laptop");

    await waitFor(() => expect(screen.getByText("kitchen.mp4")).toBeInTheDocument());
    // The video itself is on the page, sourced from the URL the API gave us.
    expect(container.querySelector("video")).toHaveAttribute(
      "src",
      "/videos/abc123/file",
    );
  });

  it("collapses several frames of one video into a single result", async () => {
    fetchMock.mockReturnValue(
      ok(
        response({
          count: 3,
          results: [
            hit({ frame_id: 1, timestamp_sec: 10 }),
            hit({ frame_id: 2, timestamp_sec: 20 }),
            hit({ frame_id: 3, timestamp_sec: 30 }),
          ],
        }),
      ),
    );
    const { container } = render(<SearchPanel />);
    await search("laptop");

    await waitFor(() => expect(screen.getByText("kitchen.mp4")).toBeInTheDocument());
    // One video, one player -- three moments inside it.
    expect(container.querySelectorAll("video")).toHaveLength(1);
    expect(screen.getByText(/3 moments/i)).toBeInTheDocument();
    expect(screen.getByText(/1 video ·/i)).toBeInTheDocument();
  });

  it("does not fire a request for a blank query", async () => {
    render(<SearchPanel />);
    const user = userEvent.setup();
    await user.type(screen.getByRole("searchbox"), "   ");

    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("says so when nothing matched, rather than going blank", async () => {
    fetchMock.mockReturnValue(ok(response({ count: 0, results: [] })));
    render(<SearchPanel />);
    await search("a giraffe on a skateboard");

    expect(await screen.findByText(/no matches/i)).toBeInTheDocument();
  });

  it("reports how long the search took", async () => {
    render(<SearchPanel />);
    await search("laptop");
    expect(await screen.findByText(/12\.5\s*ms/i)).toBeInTheDocument();
  });
});

describe("states the user can get stuck in", () => {
  it("disables the button while a search is in flight", async () => {
    let release: (value: Response) => void = () => {};
    fetchMock.mockReturnValue(new Promise<Response>((resolve) => (release = resolve)));

    render(<SearchPanel />);
    await search("laptop");

    expect(screen.getByRole("button", { name: /searching/i })).toBeDisabled();

    release({ ok: true, status: 200, json: () => Promise.resolve(response()) } as Response);
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /^search$/i })).toBeEnabled(),
    );
  });

  it("surfaces a failed request instead of showing an empty grid", async () => {
    fetchMock.mockReturnValue(
      Promise.resolve({
        ok: false,
        status: 500,
        json: () => Promise.resolve({ detail: "index is empty" }),
      } as Response),
    );

    render(<SearchPanel />);
    await search("laptop");

    expect(await screen.findByRole("alert")).toHaveTextContent(/index is empty/i);
    // An error must not read as "we searched and found nothing".
    expect(screen.queryByText(/no matches/i)).not.toBeInTheDocument();
  });

  it("surfaces a dead server, which rejects rather than returning a response", async () => {
    // Reject lazily: building the rejected promise up front makes Node flag it
    // as unhandled before the click ever awaits it.
    fetchMock.mockImplementation(() => Promise.reject(new TypeError("Failed to fetch")));

    render(<SearchPanel />);
    await search("laptop");

    expect(await screen.findByRole("alert")).toHaveTextContent(/failed to fetch/i);
  });

  it("clears a previous error once a later search succeeds", async () => {
    fetchMock.mockImplementationOnce(() => Promise.reject(new TypeError("Failed to fetch")));
    render(<SearchPanel />);
    const user = await search("laptop");
    expect(await screen.findByRole("alert")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /^search$/i }));
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
  });
});

describe("search options", () => {
  it("sends the chosen result count", async () => {
    render(<SearchPanel />);
    const user = userEvent.setup();
    await user.selectOptions(screen.getByLabelText(/results/i), "50");
    await search("laptop");

    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toMatchObject({ top_k: 50 });
  });

  it("merges near-duplicates by default", async () => {
    render(<SearchPanel />);
    await search("laptop");

    expect(
      JSON.parse(fetchMock.mock.calls[0][1].body).collapse_window_sec,
    ).toBeGreaterThan(0);
  });

  it("omits collapse_window_sec once merging is switched off", async () => {
    render(<SearchPanel />);
    const user = userEvent.setup();
    await user.click(screen.getByLabelText(/merge near-duplicates/i));
    await search("laptop");

    expect(
      JSON.parse(fetchMock.mock.calls[0][1].body).collapse_window_sec,
    ).toBeUndefined();
  });
});
