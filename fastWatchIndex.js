const config = require("./config");
const runtimeConfig = require("./runtimeConfig");
const fastWatch = require("./fastWatch");
const { sendTelegramMessage } = require("./telegram");

async function run() {
  const creds = {
    apiKey: process.env.COINDCX_API_KEY,
    apiSecret: process.env.COINDCX_API_SECRET,
  };
  if (!creds.apiKey || !creds.apiSecret) {
    throw new Error("COINDCX_API_KEY and COINDCX_API_SECRET must be set. NOTE: a read-only key is NO LONGER sufficient - fastWatch now needs write access too, to cancel an orphaned bracket order (SL or TP) once the position closes via the other side.");
  }

  // Read-only: applies whatever strategy override is CURRENTLY active
  // (as last set by the main agent's own command processing), without
  // re-processing incoming Telegram commands itself - that stays the
  // main agent's job, so there's no race between two processes both
  // trying to advance the same Telegram update-id cursor.
  const rtState = runtimeConfig.loadRuntimeConfig();
  const effectiveConfig = runtimeConfig.applyRuntimeOverrides(config, rtState);

  await fastWatch.run(effectiveConfig, creds);
}

run().catch(async (err) => {
  console.error("Fast watch fatal error:", err);
  try {
    await sendTelegramMessage(`🚨 *Fast watcher crashed*: ${err.message}`);
  } catch {
    // best-effort only
  }
  process.exit(1);
});
