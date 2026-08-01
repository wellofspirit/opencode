import { Auth } from "@/auth"
import { Provider } from "@/provider/provider"
import { ProviderTransform } from "@/provider/transform"
import { InstallationVersion } from "@opencode-ai/core/installation/version"
import { APICallError, streamText } from "ai"
import { Effect } from "effect"
import { HttpApiBuilder } from "effect/unstable/httpapi"
import os from "os"
import { InstanceHttpApi } from "../api"
import { ModelNotFoundError, UpstreamError } from "../errors"
import { JudgeCompletionPayload } from "../groups/judge"

const USER_AGENT = `opencode/${InstallationVersion}`

function modelNotFound(error: Provider.ModelNotFoundError) {
  return new ModelNotFoundError({
    providerID: error.providerID,
    modelID: error.modelID,
    suggestions: error.suggestions ?? [],
    message: error.message,
  })
}

/**
 * The AI SDK's `APICallError.message` for a 4xx is often just the HTTP reason
 * phrase ("Bad Request"), which tells a caller nothing. Carry the provider's
 * own response body — that is where the actual complaint lives.
 */
function upstream(providerID: string, cause: unknown) {
  if (APICallError.isInstance(cause)) {
    const body = typeof cause.responseBody === "string" ? cause.responseBody.slice(0, 2000) : undefined
    return new UpstreamError({
      message: body ? `${cause.message}: ${body}` : cause.message,
      service: providerID,
      ...(cause.statusCode === undefined ? {} : { status: cause.statusCode }),
    })
  }
  return new UpstreamError({
    message: cause instanceof Error ? cause.message : String(cause),
    service: providerID,
  })
}

/**
 * Provider-transport parity, replicated rather than delegated.
 *
 * The session path lets plugins reshape the request through the `chat.params`
 * and `chat.headers` hooks (session/llm/request.ts:114-146). This route cannot
 * call `Plugin.trigger` for them: both hook inputs require a real
 * `UserMessage`, and the github-copilot `chat.headers` hook dereferences
 * `message.sessionID` to fetch that session's parts
 * (plugin/github-copilot/copilot.ts:373). A judge call has no session and no
 * message, and fabricating one to satisfy arbitrary user plugins is exactly the
 * kind of session-machinery entanglement this route exists to avoid.
 *
 * So the built-in plugins' *transport* effects are replicated here, narrowly.
 * They are not optional politeness: without them a Codex/ChatGPT-OAuth judge
 * model returns a bare 400. (The single largest cause of that 400 was not a
 * hook at all — see the streaming note in the handler.)
 */

/**
 * Backends that reject an explicit output cap, so `max_output_tokens` must not
 * be sent at all. Mirrors three upstream `chat.params` hooks:
 *  - plugin/openai/codex.ts:559 — "Match codex cli"; the ChatGPT backend 400s
 *    on `max_output_tokens` at ANY value.
 *  - plugin/github-copilot/copilot.ts:344 — same, for copilot's gpt-* models.
 *  - plugin/cloudflare.ts:64 — the gateway emits `max_tokens`, which OpenAI
 *    reasoning models reject.
 */
function rejectsOutputCap(model: Provider.Model): boolean {
  if (model.providerID === "openai") return true
  if (model.providerID.includes("github-copilot") && model.api.id.includes("gpt")) return true
  if (
    model.providerID === "cloudflare-ai-gateway" &&
    model.api.id.toLowerCase().startsWith("openai/") &&
    model.capabilities.reasoning
  ) {
    return true
  }
  return false
}

/**
 * Headers the built-in plugins add. Only openai's are reproduced: the Codex
 * endpoint expects the `originator`/User-Agent shape the codex CLI sends
 * (plugin/openai/codex.ts:549-557). Copilot's extra headers are version and
 * interaction hints its API tolerates the absence of, and its hook's
 * session-parts lookup has no judge-side equivalent.
 */
function providerHeaders(model: Provider.Model, requestID: string): Record<string, string> {
  if (model.providerID !== "openai") return { "User-Agent": USER_AGENT }
  return {
    originator: "opencode",
    "User-Agent": `opencode/${InstallationVersion} (${os.platform()} ${os.release()}; ${os.arch()})`,
    "session-id": requestID,
  }
}

/**
 * `POST /judge/completion` — one system+user turn against the provider model,
 * with nothing else attached.
 *
 * This route deliberately bypasses Session, ToolRegistry and Permission. It
 * resolves the language model through `Provider.Service` and hands it to the
 * AI SDK directly: no session row is written, no tool is registered, no
 * ruleset is evaluated and no `permission.asked` can be raised. That is the
 * point — the caller is a security judge reading attacker-influenced text, so
 * "the model decided to call a tool" must not be a reachable state rather than
 * a denied one. It also keeps this patch clear of the permission layer, which
 * is mid-rewrite upstream.
 *
 * `maxTokens` and `stopSequences` are honoured for real wherever the provider
 * accepts them (the session prompt API exposes neither anywhere). `maxTokens`
 * is clamped to the model's own output ceiling. Two provider-side limits are
 * unavoidable and are handled rather than hidden: backends listed in
 * {@link rejectsOutputCap} 400 on an output cap so the field is dropped, and
 * OpenAI's Responses API has no stop-sequence parameter at all — the AI SDK
 * drops it with an "unsupported" warning (@ai-sdk/openai responses model).
 */
export const judgeHandlers = HttpApiBuilder.group(InstanceHttpApi, "judge", (handlers) =>
  Effect.gen(function* () {
    const provider = yield* Provider.Service
    const auth = yield* Auth.Service

    const completion = Effect.fn("JudgeHttpApi.completion")(function* (ctx: {
      payload: typeof JudgeCompletionPayload.Type
    }) {
      const payload = ctx.payload
      const model = yield* provider
        .getModel(payload.model.providerID, payload.model.modelID)
        .pipe(Effect.mapError(modelNotFound))
      const language = yield* provider.getLanguage(model).pipe(Effect.mapError(modelNotFound))
      const info = yield* auth.get(model.providerID).pipe(Effect.orElseSucceed(() => undefined))

      // Parity with the session path (session/llm/request.ts): the OpenAI
      // OAuth (Codex) transport carries the system prompt as `instructions`
      // rather than a system message.
      const isOpenaiOauth = model.providerID === "openai" && info?.type === "oauth"
      const options = { ...ProviderTransform.smallOptions(model) }
      if (isOpenaiOauth) options.instructions = payload.system

      const ceiling = ProviderTransform.maxOutputTokens(model)
      const maxOutputTokens = rejectsOutputCap(model)
        ? undefined
        : payload.maxTokens === undefined
          ? ceiling
          : Math.max(1, Math.min(Math.floor(payload.maxTokens), ceiling))
      const stopSequences = payload.stopSequences?.filter((x) => x.length > 0) ?? []
      const requestID = `judge-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`

      const text = yield* Effect.tryPromise({
        try: async (signal) => {
          // MUST stream. The session path always uses `streamText`, and at
          // least one provider transport depends on it: the ChatGPT/Codex
          // OAuth backend rejects a non-streaming request outright with
          // `400 {"detail":"Stream must be set to true"}`. The response is
          // still collected into one string — this route is single-turn and
          // its callers want the whole verdict, not tokens.
          let failure: unknown
          const stream = streamText({
            model: language,
            ...(isOpenaiOauth ? {} : { system: payload.system }),
            prompt: payload.user,
            ...(maxOutputTokens === undefined ? {} : { maxOutputTokens }),
            ...(stopSequences.length > 0 ? { stopSequences: [...stopSequences] } : {}),
            temperature: model.capabilities.temperature ? ProviderTransform.temperature(model) : undefined,
            topP: ProviderTransform.topP(model),
            topK: ProviderTransform.topK(model),
            providerOptions: ProviderTransform.providerOptions(model, options),
            headers: { ...model.headers, ...providerHeaders(model, requestID) },
            maxRetries: 0,
            abortSignal: signal,
            // `streamText` reports mid-stream failures here rather than
            // throwing; without this a provider error could surface as an
            // empty verdict, which a fail-closed caller must never see as
            // success.
            onError: ({ error }) => {
              failure = error
            },
          })
          const collected = await stream.text
          if (failure !== undefined) throw failure
          return collected
        },
        catch: (cause) => upstream(model.providerID, cause),
      })

      return { text }
    })

    return handlers.handle("completion", completion)
  }),
)
