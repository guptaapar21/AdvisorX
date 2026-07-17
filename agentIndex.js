const config = require("./config");
const { buildTools } = require("./agentTools");
const { generateAgentInstructions } = require("./agentInstructions");
const { runAgentCycle } = require("./geminiAgent");
const { sendTelegramMessage } = require("./telegram");

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
  });

  turnLog.forEach((line) => console.log(line));

  if (finalText) {
    console.log("Agent summary:", finalText);
    await sendTelegramMessage(`🧠 *Run summary*\n${finalText}`);
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
