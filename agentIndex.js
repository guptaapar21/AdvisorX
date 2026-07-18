const baseConfig = require("./config");
const { buildTools } = require("./agentTools");
const { generateAgentInstructions } = require("./agentInstructions");
const { runAgentCycle } = require("./geminiAgent");
const { sendTelegramMessage } = require("./telegram");
const runtimeConfig = require("./runtimeConfig");
const idleThrottle = require("./idleThrottle");

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
  let hadExecutionActivity = false;
  let hasOpenPositions = false;

  const { finalText, turnLog } = await runAgentCycle({
    userPrompt,
    systemPrompt,
    tools,
    model: config.agentModel,
    cooldownMinutes: config.geminiKeyCooldownMinutes,
    maxTurns: config.agentMaxTurns,
    onToolCall: async (name, args, result) => {
      hadExecutionActivity = true; // any execution tool firing counts as activity
      if (result.telegramMessage) {
        await sendTelegramMessage(result.telegramMessage);
      }
    },
    onReadToolResult: (name, args, result) => {
      if (name === "analyze_opening_opportunities" && result && result.allScores) {
        lastAllScores = result.allScores;
      }
      if (name === "get_positions") {
        const positions = Array.isArray(result) ? result : (result?.data || []);
        if (Array.isArray(positions) && positions.length > 0) hasOpenPositions = true;
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

    const isActive = hadExecutionActivity || hasOpenPositions;
    // Diagnostic logging - temporary, to find out why throttling isn't
    // behaving as expected. Shows the exact inputs to the decision and the
    // throttle state file's content before/after, so the Action log tells
    // us definitively what's happening instead of guessing further.
    console.log(`[throttle-debug] hadExecutionActivity=${hadExecutionActivity} hasOpenPositions=${hasOpenPositions} isActive=${isActive}`);
    console.log(`[throttle-debug] idleThrottleState.json before decision: ${idleThrottle.readRawState()}`);

    if (isActive) {
      // A position is open or a decision was made this run - message every
      // time (cron runs every 5 min), no throttling. Also reset the idle
      // timer so that once things go quiet again, the next idle message
      // fires immediately rather than waiting out a stale 15-min window.
      console.log("[throttle-debug] Taking ACTIVE branch (always sends) - resetting idle throttle.");
      idleThrottle.resetIdleThrottle();
      await sendTelegramMessage(`🧠 ${message}`);
    } else {
      const shouldSend = idleThrottle.shouldSendIdleMessage();
      console.log(`[throttle-debug] Taking IDLE branch. shouldSendIdleMessage()=${shouldSend}`);
      console.log(`[throttle-debug] idleThrottleState.json after decision: ${idleThrottle.readRawState()}`);
      if (shouldSend) {
        // Genuinely nothing going on - only send this routine update once
        // every 15 minutes even though the cron itself fires every 5.
        await sendTelegramMessage(`🧠 ${message}`);
      } else {
        console.log("Idle run - Telegram message suppressed (throttled to every 15 min while nothing is happening).");
      }
    }
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
