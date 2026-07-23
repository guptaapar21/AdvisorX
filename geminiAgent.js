const { withKeyRotation } = require("./geminiKeys");
const { parseQuotaInfo } = require("./geminiQuotaInfo");

async function callGemini(contents, systemPrompt, toolDeclarations, apiKey, model) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-goog-api-key": apiKey },
    body: JSON.stringify({
      system_instruction: { parts: [{ text: systemPrompt }] },
      tools: [{ functionDeclarations: toolDeclarations }],
      contents,
    }),
  });

  if (res.status === 429) {
    // Read Google's ACTUAL quota details instead of guessing a flat
    // cooldown - this is what previously made a genuine per-day quota hit
    // look like it "keeps happening all day": every key was cooling down
    // for only 60 minutes, retrying, instantly hitting the same still-
    // exhausted daily cap, and cooling down again, on a loop that only a
    // real midnight-Pacific reset (not the 60-minute timer) could break.
    const { period, retryDelayMs, quotaId } = await parseQuotaInfo(res);
    const err = new Error(
      `Gemini quota/rate limit exceeded${quotaId ? ` (${quotaId})` : ""}`
    );
    err.rateLimited = true;
    err.transientReason = "quota_exceeded";
    err.quotaPeriod = period; // "day" | "short" | "unknown"
    err.retryDelayMs = retryDelayMs; // Google's own suggested wait, if given
    throw err;
  }
  if (res.status === 503) {
    // "Model currently experiencing high demand" - this is genuinely
    // temporary and recoverable, just like a 429, NOT a code/config bug.
    // Was previously falling into the generic error bucket below and
    // getting wrongly diagnosed as "needs a code fix, waiting won't help"
    // - a real diagnosis mistake found from a live failure message.
    const err = new Error("Gemini model temporarily overloaded (503)");
    err.rateLimited = true;
    err.transientReason = "model_overloaded";
    throw err;
  }
  if (res.status === 404) {
    const body = await res.text();
    throw new Error(
      `Gemini call failed: 404 (model "${model}" not found/unavailable - Google renames these periodically; ` +
      `set the GEMINI_MODEL secret to override without a code change, e.g. gemini-3.5-flash or gemini-flash-latest) ${body}`
    );
  }
  if (!res.ok) {
    throw new Error(`Gemini call failed: ${res.status} ${await res.text()}`);
  }
  return res.json();
}

// Runs the agent for one full reasoning cycle: repeatedly calls Gemini,
// executes whatever tools it asks for (via `tools.handlers`), and feeds the
// results back, until the model responds with plain text and no more tool
// calls (or `maxTurns` is hit as a safety cap). `onToolCall` is invoked for
// every tool call so the caller can log/relay execution-tool messages.
async function runAgentCycle({ userPrompt, systemPrompt, tools, model, cooldownMinutes, maxTurns, onToolCall, onReadToolResult }) {
  const contents = [{ role: "user", parts: [{ text: userPrompt }] }];
  const turnLog = [];

  for (let turn = 0; turn < (maxTurns || 8); turn++) {
    const { result: response, diagnosis, details, keyCount, skippedInCooldown } = await withKeyRotation(
      (key) => callGemini(contents, systemPrompt, tools.declarations, key, model),
      cooldownMinutes
    );

    if (!response) {
      // Real diagnosis instead of a vague "no key available" - tells the
      // difference between "genuinely all out of quota" (expected,
      // recoverable) and "something is actually broken" (won't fix itself).
      let diagnosisText;
      if (diagnosis === "no_keys_configured") {
        diagnosisText = "no Gemini keys configured at all - check GEMINI_API_KEYS/GEMINI_API_KEY_n secrets";
      } else if (diagnosis === "all_keys_in_cooldown_from_earlier") {
        diagnosisText = `all ${keyCount} key(s) still in cooldown from an earlier hit this run - none were even attempted`;
      } else if (diagnosis === "all_keys_quota_exhausted") {
        const reasons = new Set(details.map((d) => d.transientReason));
        let reasonText;
        if (reasons.size === 1 && reasons.has("model_overloaded")) {
          reasonText = "Google's model is temporarily overloaded (503, high demand) - NOT your quota, just try again shortly";
        } else if (reasons.size === 1 && reasons.has("quota_exceeded")) {
          const periods = new Set(details.map((d) => d.quotaPeriod));
          if (periods.size === 1 && periods.has("day")) {
            reasonText = "genuinely hit the DAILY request quota (429, PerDay) on every key - this only resets at midnight Pacific time, not within the hour. Consider fewer Gemini calls per run, a longer run interval, or more/higher-tier keys";
          } else if (periods.size === 1 && periods.has("short")) {
            reasonText = "hit a short-lived per-minute/per-second limit (429) on every key - recovers within seconds to a couple minutes, no action needed";
          } else {
            reasonText = "hit quota limits (429) - real exhaustion, will recover at reset";
          }
        } else {
          reasonText = "a mix of quota limits (429) and temporary model overload (503) - both recover on their own, no fix needed";
        }
        diagnosisText = `all ${keyCount} key(s) hit a transient limit this run - ${reasonText}`;
      } else if (diagnosis === "genuine_errors_not_quota") {
        const sample = details.find((d) => d.type === "error");
        diagnosisText = `NOT transient - at least one key failed with a real error: "${sample?.message}" - this needs a code/config fix, waiting won't help`;
      } else {
        diagnosisText = "unknown - see raw Action log for per-key details";
      }
      turnLog.push(`(no Gemini key available this run - ${diagnosisText})`);
      return { finalText: null, turnLog };
    }

    const candidate = response.candidates?.[0];
    const parts = candidate?.content?.parts || [];
    const functionCalls = parts.filter((p) => p.functionCall).map((p) => p.functionCall);
    const textParts = parts.filter((p) => p.text).map((p) => p.text);

    if (functionCalls.length === 0) {
      const finalText = textParts.join("\n").trim() || null;
      return { finalText, turnLog };
    }

    // Record the model's turn (including its function call requests)
    contents.push({ role: "model", parts });

    const functionResponseParts = [];
    for (const call of functionCalls) {
      const handler = tools.handlers[call.name];
      let resultForModel;

      if (!handler) {
        resultForModel = { error: `unknown tool: ${call.name}` };
      } else {
        try {
          const result = await handler(call.args || {});
          if (tools.isExecutionTool(call.name)) {
            if (onToolCall) await onToolCall(call.name, call.args, result);
            resultForModel = result.resultForModel;
            turnLog.push(`[execution] ${call.name}(${JSON.stringify(call.args)})`);
          } else {
            if (onReadToolResult) onReadToolResult(call.name, call.args, result);
            resultForModel = result;
            turnLog.push(`[read] ${call.name}(${JSON.stringify(call.args)})`);
          }
        } catch (err) {
          resultForModel = { error: err.message };
          turnLog.push(`[error] ${call.name}: ${err.message}`);
        }
      }

      // Gemini's functionResponse.response field must be a JSON object,
      // not a bare array or scalar - several of our read tools (get_positions,
      // get_account_balance) return raw arrays straight from CoinDCX, so wrap
      // anything that isn't already a plain object.
      const isPlainObject = resultForModel != null && typeof resultForModel === "object" && !Array.isArray(resultForModel);
      const wrappedResponse = isPlainObject ? resultForModel : { result: resultForModel ?? null };

      functionResponseParts.push({
        functionResponse: { name: call.name, response: wrappedResponse },
      });
    }

    contents.push({ role: "user", parts: functionResponseParts });
  }

  turnLog.push(`(hit max turns (${maxTurns}) without a final answer)`);
  return { finalText: null, turnLog };
}

module.exports = { runAgentCycle };
