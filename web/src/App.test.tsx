/**
 * What the search page must do (M5).
 *
 * The backend is stubbed at `fetch`: these are tests of the UI's behaviour, not
 * of the API, which has its own suite. What matters here is that a query
 * reaches the right endpoint in the right shape, that every state the user can
 * land in says something, and that no state leaves the page silently blank.
 */
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type { SearchResponse } from "./api";

const hit = (over: Partial<SearchResponse["results"][0]> = {}) => ({
  score: 0.31,
  video_id: "abc123",
  filename: "kitchen.mp4",
  timestamp_sec: 83.5,
  thumbnail_url: "/thumbnails/42",
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
  await user.click(screen.getByRole("button", { name: /search/i }));
  return user;
};

describe("searching", () => {
  it("posts the query to /search and renders the hits", async () => {
    render(<App />);
    await search("a person opens a laptop");

    await waitFor(() => expect(screen.getByText("kitchen.mp4")).toBeInTheDocument());

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/search");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toMatchObject({ query: "a person opens a laptop" });
  });

  it("shows each hit's timestamp and thumbnail", async () => {
    render(<App />);
    await search("laptop");

    const thumb = await screen.findByRole("img", { name: /kitchen\.mp4 at 1:23/i });
    // The URL comes from the API verbatim. Rebuilding it in the client would
    // duplicate a decision the server already owns.
    expect(thumb).toHaveAttribute("src", "/thumbnails/42");
    expect(screen.getByText("1:23")).toBeInTheDocument();
  });

  it("does not fire a request for a blank query", async () => {
    render(<App />);
    const user = userEvent.setup();
    await user.type(screen.getByRole("searchbox"), "   ");
    await user.click(screen.getByRole("button", { name: /search/i }));

    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("says so when nothing matched, rather than going blank", async () => {
    fetchMock.mockReturnValue(ok(response({ count: 0, results: [] })));
    render(<App />);
    await search("a giraffe on a skateboard");

    expect(await screen.findByText(/no matches/i)).toBeInTheDocument();
  });

  it("reports how long the search took", async () => {
    render(<App />);
    await search("laptop");
    expect(await screen.findByText(/12\.5\s*ms/i)).toBeInTheDocument();
  });
});

describe("states the user can get stuck in", () => {
  it("disables the button while a search is in flight", async () => {
    let release: (value: Response) => void = () => {};
    fetchMock.mockReturnValue(new Promise<Response>((resolve) => (release = resolve)));

    render(<App />);
    await search("laptop");

    const button = screen.getByRole("button", { name: /searching/i });
    expect(button).toBeDisabled();

    release({ ok: true, status: 200, json: () => Promise.resolve(response()) } as Response);
    await waitFor(() => expect(screen.getByRole("button", { name: /search/i })).toBeEnabled());
  });

  it("surfaces a failed request instead of showing an empty grid", async () => {
    fetchMock.mockReturnValue(
      Promise.resolve({
        ok: false,
        status: 500,
        json: () => Promise.resolve({ detail: "index is empty" }),
      } as Response),
    );

    render(<App />);
    await search("laptop");

    const error = await screen.findByRole("alert");
    expect(error).toHaveTextContent(/index is empty/i);
    // An error must not read as "we searched and found nothing".
    expect(screen.queryByText(/no matches/i)).not.toBeInTheDocument();
  });

  it("surfaces a dead server, which rejects rather than returning a response", async () => {
    // Reject lazily: building the rejected promise up front makes Node flag it
    // as unhandled before the click ever awaits it.
    fetchMock.mockImplementation(() => Promise.reject(new TypeError("Failed to fetch")));

    render(<App />);
    await search("laptop");

    expect(await screen.findByRole("alert")).toHaveTextContent(/failed to fetch/i);
  });

  it("clears a previous error once a later search succeeds", async () => {
    fetchMock.mockImplementationOnce(() => Promise.reject(new TypeError("Failed to fetch")));
    render(<App />);
    const user = await search("laptop");
    expect(await screen.findByRole("alert")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /search/i }));
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
  });
});

describe("search options", () => {
  it("sends the chosen result count", async () => {
    render(<App />);
    const user = userEvent.setup();
    await user.selectOptions(screen.getByLabelText(/results/i), "25");
    await search("laptop");

    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toMatchObject({ top_k: 25 });
  });

  it("omits collapse_window_sec unless collapsing is asked for", async () => {
    render(<App />);
    await search("laptop");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body).collapse_window_sec).toBeUndefined();
  });

  it("sends collapse_window_sec when near-duplicate merging is on", async () => {
    render(<App />);
    const user = userEvent.setup();
    await user.click(screen.getByLabelText(/merge near-duplicates/i));
    await search("laptop");

    expect(JSON.parse(fetchMock.mock.calls[0][1].body).collapse_window_sec).toBeGreaterThan(0);
  });
});
