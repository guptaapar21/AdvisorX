const { withKeyRotation } = require("./geminiKeys");

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
    const err = new Error("Gemini quota/rate limit exceeded");
    err.rateLimited = true;
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
async function runAgentCycle({ userPrompt, systemPrompt, tools, model, cooldownMinutes, maxTurns, onToolCall }) {
  const contents = [{ role: "user", parts: [{ text: userPrompt }] }];
  const turnLog = [];

  for (let turn = 0; turn < (maxTurns || 8); turn++) {
    const response = await withKeyRotation(
      (key) => callGemini(contents, systemPrompt, tools.declarations, key, model),
      cooldownMinutes
    );

    if (!response) {
      turnLog.push("(no Gemini key available this run - stopped)");
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
