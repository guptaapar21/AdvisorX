const baseConfig = require("./config");
const { buildTools } = require("./agentTools");
const { generateAgentInstructions } = require("./agentInstructions");
const { runAgentCycle } = require("./geminiAgent");
const { sendTelegramMessage } = require("./telegram");
const runtimeConfig = require("./runtimeConfig");

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
  // rejection reply itself, so this never fails silently.
  let rtState = runtimeConfig.loadRuntimeConfig();
  rtState = await runtimeConfig.processIncomingCommands(rtState);
  const config = runtimeConfig.applyRuntimeOverrides(baseConfig, rtState);
  runtimeConfig.saveRuntimeConfig(rtState);

  const tools = buildTools(config, creds);
  const systemPrompt = generateAgentInstructions(config);
  const userPrompt = [
    `Begin your reasoning cycle for this run.`,
    `Configured symbols: ${config.symbols.join(", ")}`,
    `Market type: ${config.marketType} | Strategy: ${config.strategy} | Timeframes: primary ${config.timeframes.primary} / confirm ${config.timeframes.confirm} / filter ${config.timeframes.filter}`,
    `Follow the decision priority in your instructions: manage existing positions first, then look for new entries.`,
  ].join("\n");

  console.log("Starting agent reasoning cycle...");

  let lastAllScores = null;

  const { finalText, turnLog } = await runAgentCycle({
    userPrompt,
    systemPrompt,
    tools,
    model: config.agentModel,
    cooldownMinutes: config.geminiKeyCooldownMinutes,
    maxTurns: config.agentMaxTurns,
    onToolCall: async (name, args, result) => {
      if (result.telegramMessage) {
        await sendTelegramMessage(result.telegramMessage);
      }
    },
    onReadToolResult: (name, args, result) => {
      if (name === "analyze_opening_opportunities" && result && result.allScores) {
        lastAllScores = result.allScores;
      }
    },
  });

  turnLog.forEach((line) => console.log(line));

  const warnings = tools.getWarnings();

  if (finalText) {
    console.log("Agent summary:", finalText);
    let message = crispSummary(finalText);
    // If this run scanned for new opportunities, always append the real
    // per-symbol scores ourselves - don't rely on the model to have
    // included them in its free-text reason line.
    const scoresLine = formatScoresLine(lastAllScores);
    if (scoresLine && !message.includes("Scores:")) {
      message = `${message}\n${scoresLine}`;
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
    await sendTelegramMessage(
      `🚨 *Agent run failed* - no response from Gemini this cycle (likely all keys exhausted or erroring).\n` +
      `Last log lines: ${lastLines || "none"}\n` +
      `Check the Action logs for details. No decisions were made this run.`
    );
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
