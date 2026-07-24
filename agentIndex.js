const baseConfig = require("./config");
const { buildTools } = require("./agentTools");
const { generateAgentInstructions } = require("./agentInstructions");
const { runAgentCycle } = require("./geminiAgent");
const { sendTelegramMessage } = require("./telegram");
const runtimeConfig = require("./runtimeConfig");
const idleThrottle = require("./idleThrottle");
const { runPreFilter } = require("./preFilter");

function crispSummary(text) {
  // Safety net: keep the run-log message short even if the model doesn't
  // follow the requested two-line format exactly. Keep at most the status
  // line plus one reason line, each hard-capped in length.
  const lines = text.split("\n").map((l) => l.trim()).filter((l) => l.length > 0);
  const cap = (l, n) => (l.length > n ? `${l.slice(0, n - 3)}...` : l);
  const statusLine = lines[0] ? cap(lines[0], 100) : "";
  const reasonLine = lines[1] ? cap(lines[1], 140) : "";
  return [statusLine, reasonLine].filter(Boolean).join("\n");
}

// Builds "Reversal detail (XRP): primary timeframe entering range (score=-8)"
// deterministically from check_reversal's own returned `details` array,
// same reasoning as formatScoresLine below - Gemini was ASKED (in
// agentInstructions.js) to cite the underlying raw trend number alongside
// the bucketed reversalScore, so the "12" doesn't look frozen run to run,
// but it doesn't reliably do so (same class of drift as the contract-
// string issue). Building it in code guarantees it's always there.
function formatReversalDetailLines(reversalDetailsBySymbol) {
  const symbols = Object.keys(reversalDetailsBySymbol);
  if (symbols.length === 0) return null;
  return symbols
    .map((sym) => {
      const d = reversalDetailsBySymbol[sym];
      const detailText = d.details && d.details.length > 0 ? d.details.join("; ") : "no active drivers this cycle";
      return `Reversal detail (${sym}): score ${d.reversalScore} - ${detailText}`;
    })
    .join("\n");
}

// Builds "Scores: BTC 45, ETH 58, SOL 73, ..." deterministically from the
// real allScores data, rather than trusting the model to always include
// every symbol in its free-text reasoning.
function formatScoresLine(allScores) {
  if (!allScores || allScores.length === 0) return null;
  const parts = allScores
    .slice()
    .sort((a, b) => b.score - a.score)
    .map((s) => `${s.symbol} ${s.score}`);
  return `Scores: ${parts.join(", ")}`;
}

// Builds the routine "nothing to do" message directly from the pre-filter's
// own real data - no Gemini call needed for this, the common case.
function formatIdleMessage(preFilterResult, config) {
  const scored = preFilterResult.allScores.filter((s) => typeof s.score === "number");
  scored.sort((a, b) => b.score - a.score);
  const best = scored[0];

  const statusLine = "NO ACTION — no positions, no setups cleared threshold.";
  const reasonLine = best
    ? `Reason: Best candidate ${best.symbol} scored ${best.score}${best.setupType ? ` (${best.setupType})` : ""}, below the required ${config.minScore}.`
    : "Reason: No valid candidates this run.";
  const scoresLine = formatScoresLine(preFilterResult.allScores);

  return [statusLine, reasonLine, scoresLine].filter(Boolean).join("\n");
}

async function run() {
  const creds = {
    apiKey: process.env.COINDCX_API_KEY,
    apiSecret: process.env.COINDCX_API_SECRET,
  };
  if (!creds.apiKey || !creds.apiSecret) {
    throw new Error("COINDCX_API_KEY and COINDCX_API_SECRET must be set (read-only key is sufficient)");
  }

  // Check for any Telegram command (e.g. "/strategy aggressive") sent
  // since the last run, and apply it. Always sends a confirmation/
  // rejection reply itself, so this never fails silently. Cheap - no
  // Gemini involved, just a Telegram API call.
  let rtState = runtimeConfig.loadRuntimeConfig();
  rtState = await runtimeConfig.processIncomingCommands(rtState);
  const config = runtimeConfig.applyRuntimeOverrides(baseConfig, rtState);
  runtimeConfig.saveRuntimeConfig(rtState);

  console.log("Running pre-filter (no Gemini calls)...");
  const preFilterResult = await runPreFilter(config, creds);
  console.log(
    `Pre-filter result: hasOpenPositions=${preFilterResult.hasOpenPositions}, ` +
    `qualifyingCandidates=${preFilterResult.candidates.length}, isActive=${preFilterResult.isActive}`
  );

  if (!preFilterResult.isActive) {
    // Nothing worth spending a Gemini call on this run - skip the agent
    // entirely and just report the pre-filter's own real findings,
    // throttled to once every 15 min like before.
    console.log("Skipping Gemini this run - pre-filter found nothing active.");
    const message = formatIdleMessage(preFilterResult, config);
    if (idleThrottle.shouldSendIdleMessage()) {
      await sendTelegramMessage(`🧠 ${message}`);
    } else {
      console.log("Idle run - Telegram message suppressed (throttled to every 15 min while nothing is happening).");
    }
    console.log("Run complete (pre-filter only, no Gemini used).");
    return;
  }

  // Something's active (a real position, or a candidate cleared minScore) -
  // worth spending a Gemini call to actually reason about it.
  console.log("Pre-filter found activity - invoking the full Gemini agent...");
  idleThrottle.resetIdleThrottle();

  const tools = buildTools(config, creds);
  const systemPrompt = generateAgentInstructions(config);
  const userPrompt = [
    `Begin your reasoning cycle for this run.`,
    `Configured symbols: ${config.symbols.join(", ")}`,
    `Market type: ${config.marketType} | Strategy: ${config.strategy} | Timeframes: primary ${config.timeframes.primary} / confirm ${config.timeframes.confirm} / filter ${config.timeframes.filter}`,
    `Follow the decision priority in your instructions: manage existing positions first, then look for new entries.`,
  ].join("\n");

  console.log("Starting agent reasoning cycle...");

  let lastAllScores = preFilterResult.allScores;
  let hadExecutionThisRun = false;
  const reversalDetailsBySymbol = {};

  const { finalText, turnLog } = await runAgentCycle({
    userPrompt,
    systemPrompt,
    tools,
    model: config.agentModel,
    cooldownMinutes: config.geminiKeyCooldownMinutes,
    maxTurns: config.agentMaxTurns,
    onToolCall: async (name, args, result) => {
      if (result.telegramMessage) {
        hadExecutionThisRun = true;
        await sendTelegramMessage(result.telegramMessage);
      }
    },
    onReadToolResult: (name, args, result) => {
      if (name === "analyze_opening_opportunities" && result && result.allScores) {
        lastAllScores = result.allScores;
      }
      if (name === "check_reversal" && args && args.symbol && result && typeof result.reversalScore === "number") {
        reversalDetailsBySymbol[args.symbol] = result;
      }
    },
  });

  turnLog.forEach((line) => console.log(line));

  const warnings = tools.getWarnings();

  if (finalText) {
    console.log("Agent summary:", finalText);
    let message = crispSummary(finalText);
    const scoresLine = formatScoresLine(lastAllScores);
    if (scoresLine && !message.includes("Scores:")) {
      message = `${message}\n${scoresLine}`;
    }
    const reversalLines = formatReversalDetailLines(reversalDetailsBySymbol);
    if (reversalLines) {
      message = `${message}\n${reversalLines}`;
    }
    if (warnings.length > 0) {
      message = `${message}\n⚠️ ${warnings.length} issue(s) this run: ${warnings.join(" | ")}`;
    }
    await sendTelegramMessage(`🧠 ${message}`);
  } else {
    // The model never produced a final answer - almost always means every
    // configured Gemini key failed/exhausted this run. This MUST be
    // visible on Telegram, not just in the Action logs, or a silent
    // failure looks identical to "nothing to do".
    console.log("Agent produced no final text this run (see turn log above).");
    const lastLines = turnLog.slice(-3).join(" | ");
    if (hadExecutionThisRun) {
      // An execution tool already fired and sent its own message earlier
      // in this same run - the action ABOVE is real and already sent, this
      // alert is only about the wrap-up summary failing, not about nothing
      // having happened. Saying "no decisions were made" here would be
      // straightforwardly wrong.
      await sendTelegramMessage(
        `🚨 *Note*: the action above was sent successfully, but Gemini ran out of keys/quota before it could finish this run's final summary.\n` +
        `Last log lines: ${lastLines || "none"}\n` +
        `Check the Action logs for details - the action above is real and stands on its own.`
      );
    } else {
      await sendTelegramMessage(
        `🚨 *Agent run failed* - no response from Gemini this cycle. Real diagnosis below (not just "exhausted or erroring" guesswork):\n` +
        `${lastLines || "none"}\n` +
        `Check the Action logs for full per-key details. No decisions were made this run.`
      );
    }
  }

  tools.persistAdvisories();
  console.log("Run complete.");
}

run().catch(async (err) => {
  console.error("Fatal error:", err);
  // Best-effort: try to alert on Telegram too, since a console-only error
  // is invisible unless she's actively checking the Action logs.
  try {
    await sendTelegramMessage(`🚨 *Agent crashed*: ${err.message}\nCheck the Action logs for the full stack trace.`);
  } catch {
    // If Telegram itself isn't configured/reachable, there's nothing more
    // we can do here - the Action log and non-zero exit code are the
    // remaining signal.
  }
  process.exit(1);
});
