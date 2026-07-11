import { describe, expect, it } from "vitest";
import { formatTimestamp } from "./format";

describe("formatTimestamp", () => {
  it("renders seconds as m:ss so a timestamp is scannable", () => {
    expect(formatTimestamp(0)).toBe("0:00");
    expect(formatTimestamp(7.2)).toBe("0:07");
    expect(formatTimestamp(83.5)).toBe("1:23");
  });

  it("pads past the hour rather than wrapping back to zero", () => {
    expect(formatTimestamp(3661)).toBe("1:01:01");
  });

  it("floors rather than rounds, so a hit never points past itself", () => {
    // 59.9s rounded would read 1:00 -- a second later than the frame we have.
    expect(formatTimestamp(59.9)).toBe("0:59");
  });
});
