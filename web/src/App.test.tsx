/** The shell: tabs, and the panels they reveal. Panel behaviour is tested in
 *  each panel's own suite. */
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn(() =>
      Promise.resolve({
        ok: true,
        status: 200,
        json: () =>
          Promise.resolve({
            status: "ok",
            videos: 5,
            frames: 87,
            vectors: 87,
            device: "mps",
            // /videos is polled by the library tab.
            videos_list: [],
          }),
      } as Response),
    ),
  );
});

afterEach(() => vi.unstubAllGlobals());

describe("App shell", () => {
  it("opens on the search tab", async () => {
    render(<App />);
    expect(screen.getByRole("searchbox")).toBeInTheDocument();
    // Settle the health fetch before unmount, so its state update does not
    // land outside act() and warn.
    await screen.findByText(/5 videos/i);
  });

  it("offers search, library, how-to-use and about", async () => {
    render(<App />);
    for (const name of [/search/i, /library/i, /how to use/i, /about/i]) {
      expect(screen.getByRole("tab", { name })).toBeInTheDocument();
    }
    await screen.findByText(/5 videos/i);
  });

  it("switches panels when a tab is chosen", async () => {
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText(/5 videos/i);
    await user.click(screen.getByRole("tab", { name: /how to use/i }));
    expect(
      await screen.findByText(/describe what is on screen/i),
    ).toBeInTheDocument();
    expect(screen.queryByRole("searchbox")).not.toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: /about/i }));
    expect(await screen.findByText(/how it works/i)).toBeInTheDocument();
  });

  it("shows corpus size once health answers", async () => {
    render(<App />);
    expect(await screen.findByText(/5 videos · 87 frames/i)).toBeInTheDocument();
  });

  it("still renders when health is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new TypeError("nope"))));
    render(<App />);
    // The counters are context, not function: losing them must not take the
    // page down or raise an alarm the user cannot act on.
    expect(screen.getByRole("searchbox")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
