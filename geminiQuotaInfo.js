// Parses the ACTUAL quota info Gemini puts in a 429 response body, instead
// of guessing. This is the real fix for keys looking "exhausted all day":
// Google's 429 body includes a QuotaFailure detail naming which quota was
// hit (quotaId - contains "PerDay" for a daily cap that only resets at
// midnight Pacific, or "PerMinute"/"PerSecond" for a short-lived cap) and a
// RetryInfo detail with the real retryDelay (e.g. "14s"). The old code never
// read the body at all and always waited a flat configured number of
// minutes - fine for a per-minute limit, wildly wrong for a per-day one:
// that cooldown expires long before the daily quota resets, so the key gets
// retried, instantly 429s again, and re-cools down, endlessly, all day.
//
// Call this with the Response object as soon as you see status 429 (before
// anything else consumes the body).
async function parseQuotaInfo(res) {
  let bodyText = "";
  try {
    bodyText = await res.text();
  } catch {
    return { period: "unknown", retryDelayMs: null, raw: null };
  }

  let parsed = null;
  try {
    parsed = JSON.parse(bodyText);
  } catch {
    // Non-JSON body - fall through with what we have.
  }

  const details = parsed?.error?.details || [];

  const quotaFailure = details.find(
    (d) => d["@type"] === "type.googleapis.com/google.rpc.QuotaFailure"
  );
  const quotaId = quotaFailure?.violations?.[0]?.quotaId || "";

  const retryInfo = details.find(
    (d) => d["@type"] === "type.googleapis.com/google.rpc.RetryInfo"
  );
  const retryDelayStr = retryInfo?.retryDelay; // e.g. "14s" or "34.07s"
  let retryDelayMs = null;
  if (typeof retryDelayStr === "string") {
    const seconds = parseFloat(retryDelayStr.replace(/s$/, ""));
    if (!Number.isNaN(seconds)) retryDelayMs = Math.round(seconds * 1000);
  }

  let period;
  if (/PerDay/i.test(quotaId)) {
    period = "day";
  } else if (/Per(Minute|Second|Hour)/i.test(quotaId)) {
    period = "short";
  } else if (retryDelayMs !== null) {
    // No quotaId, but a real retryDelay was given - trust that directly
    // rather than assume it's a day-long cap.
    period = "short";
  } else {
    period = "unknown";
  }

  return { period, retryDelayMs, quotaId: quotaId || null, raw: bodyText };
}

// Milliseconds until the next Gemini free-tier quota reset, which happens
// at midnight US Pacific time (confirmed in Google's own rate-limit docs).
// Used as the cooldown for a confirmed per-day quota hit, instead of the
// flat configured minutes that has nothing to do with when the quota
// actually clears.
function msUntilNextPacificMidnight(now = Date.now()) {
  const pacificNowStr = new Date(now).toLocaleString("en-US", { timeZone: "America/Los_Angeles" });
  const pacificNow = new Date(pacificNowStr);
  const nextMidnightPacific = new Date(pacificNow);
  nextMidnightPacific.setHours(24, 0, 0, 0); // rolls to next-day 00:00
  const diffMs = nextMidnightPacific.getTime() - pacificNow.getTime();
  // Small safety buffer so we don't retry a few seconds before the actual
  // reset lands and immediately re-cooldown for another full day.
  return diffMs + 2 * 60 * 1000;
}

module.exports = { parseQuotaInfo, msUntilNextPacificMidnight };
