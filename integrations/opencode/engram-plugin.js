/**
 * Engram Memory Plugin for OpenCode
 *
 * Automatic memory integration: injects context before every LLM call
 * and logs + learns after every response. No manual tool calls needed.
 *
 * Installation:
 *   1. Copy this file to ~/.config/opencode/plugins/engram-plugin.js
 *   2. Add engram as an MCP server in opencode.json:
 *      "mcpServers": {
 *        "engram": {
 *          "command": "engram",
 *          "args": ["serve", "--data-dir", "/path/to/memory"]
 *        }
 *      }
 *   3. Restart OpenCode
 *
 * The plugin hooks into:
 *   - session.created  -> calls engram_boot to ground identity
 *   - message.updated  -> after each assistant response, calls engram_after
 *   - tool.execute.after -> captures trace IDs from engram_before calls
 *
 * The SKILL.md handles the before-pipeline (the model calls engram_before
 * voluntarily because the skill instructs it to). This plugin handles the
 * boot and after-pipeline automatically so nothing is ever missed.
 */

// Plugin state
let initialized = false;
let currentPerson = "unknown";
let currentSource = "opencode";
let lastUserMessage = "";
let lastTraceIds = [];
let exchangeCount = 0;

/**
 * Engram Memory Plugin
 */
export const EngramPlugin = async ({ client, project }) => {
  return {
    /**
     * Hook: Session Created
     *
     * Boot memory system and load initial context.
     */
    "session.created": async () => {
      try {
        // Boot the memory system — loads identity, top memories,
        // preferences, boundaries, anchoring beliefs, active injuries,
        // and recent journal entries. This grounds identity coherence
        // at the start of every session.
        const bootData = await client.tools.execute(
          "mcp-engram-engram_boot",
          {},
        );

        initialized = true;
        exchangeCount = 0;

        // Try to determine who we're talking to from the project context
        // Default to "developer" for coding sessions
        currentPerson = "developer";

      } catch (error) {
        // Engram MCP not available - degrade gracefully
        initialized = false;
      }
    },

    /**
     * Hook: Message Updated
     *
     * Track user messages and auto-log assistant responses.
     */
    "message.updated": async ({ message }) => {
      if (!initialized) return;

      // Track user messages for the after-pipeline
      if (message.role === "user") {
        lastUserMessage = message.content || "";
        return;
      }

      // Only process completed assistant messages
      if (message.role !== "assistant") return;
      if (!message.content) return;

      const response = message.content;

      // Don't log very short responses (likely tool-use fragments)
      if (response.length < 20) return;

      try {
        // Call engram_after to log the exchange and learn
        const result = await client.tools.execute(
          "mcp-engram-engram_after",
          {
            person: currentPerson,
            their_message: lastUserMessage || "[session interaction]",
            response: response,
            source: currentSource,
            trace_ids: lastTraceIds.join(","),
          },
        );

        exchangeCount++;

        // Parse result to extract signal info
        if (typeof result === "string") {
          try {
            const parsed = JSON.parse(result);
            // Store trace IDs for next reinforcement cycle
            if (parsed.logged_trace_id) {
              lastTraceIds = [parsed.logged_trace_id];
            }
          } catch (e) {
            // Not JSON, that's fine
          }
        }

      } catch (error) {
        // Silent fail - don't interrupt the user's workflow
      }
    },

    /**
     * Hook: Tool Execute Before
     *
     * Intercept engram_before calls to capture trace IDs.
     */
    "tool.execute.before": async (input, output) => {
      // No-op: we let the model call engram_before via the SKILL
    },

    /**
     * Hook: Tool Execute After
     *
     * Capture trace IDs from engram_before responses.
     */
    "tool.execute.after": async (input, output) => {
      if (!initialized) return;

      try {
        // Check if this was an engram_before call
        const toolName = input?.name || input?.tool || "";
        if (toolName.includes("engram_before")) {
          // Extract trace IDs and person from the result
          const resultStr = typeof output === "string" ? output : JSON.stringify(output);
          const parsed = JSON.parse(resultStr);

          if (parsed.trace_ids) {
            lastTraceIds = parsed.trace_ids;
          }
          if (parsed.person) {
            currentPerson = parsed.person;
          }
        }
      } catch (e) {
        // Silent
      }
    },

    /**
     * Custom Tool: engram_status
     *
     * Quick status check for the memory system.
     */
    tool: {
      engram_status: {
        description: "Check Engram memory system status",
        args: {},
        async execute(args, context) {
          if (!initialized) {
            return {
              initialized: false,
              message: "Engram not connected. Ensure the MCP server is running.",
            };
          }

          try {
            const stats = await client.tools.execute(
              "mcp-engram-engram_stats",
              {},
            );
            const signal = await client.tools.execute(
              "mcp-engram-engram_signal",
              {},
            );

            return {
              initialized: true,
              exchanges_this_session: exchangeCount,
              current_person: currentPerson,
              stats: typeof stats === "string" ? JSON.parse(stats) : stats,
              signal: typeof signal === "string" ? JSON.parse(signal) : signal,
            };
          } catch (error) {
            return {
              initialized: true,
              exchanges_this_session: exchangeCount,
              error: error.message || "Failed to get stats",
            };
          }
        },
      },
    },
  };
};

export default EngramPlugin;
