const config = require("./config");
const fastWatch = require("./fastWatch");
const { sendTelegramMessage } = require("./telegram");

async function run() {
  const creds = {
    apiKey: process.env.COINDCX_API_KEY,
    apiSecret: process.env.COINDCX_API_SECRET,
  };
  if (!creds.apiKey || !creds.apiSecret) {
    throw new Error("COINDCX_API_KEY and COINDCX_API_SECRET must be set (read-only key is sufficient)");
  }

  await fastWatch.run(config, creds);
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
