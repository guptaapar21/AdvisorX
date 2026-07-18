const config = require("./config");
const { buildTools } = require("./agentTools");
const { generateAgentInstructions } = require("./agentInstructions");
const { runAgentCycle } = require("./geminiAgent");
const { sendTelegramMessage } = require("./telegram");

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

  const tools = buildTools(config, creds);
  const systemPrompt = generateAgentInstructions(config);
  const userPrompt = [
    `Begin your reasoning cycle for this run.`,
    `Configured symbols: ${config.symbols.join(", ")}`,
    `Market type: ${config.marketType} | Entry timeframe: ${config.entryTimeframe} | Trend timeframe: ${config.trendTimeframe}`,
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
    await sendTelegramMessage(`🧠 ${message}`);
  } else {
    console.log("Agent produced no final text this run (see turn log above).");
  }

  tools.persistAdvisories();
  console.log("Run complete.");
}

run().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
