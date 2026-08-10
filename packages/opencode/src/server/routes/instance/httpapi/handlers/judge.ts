import { Auth } from "@/auth"
import { Provider } from "@/provider/provider"
import { ProviderTransform } from "@/provider/transform"
import { InstallationVersion } from "@opencode-ai/core/installation/version"
import { APICallError, streamText, type ModelMessage } from "ai"
import { createHash } from "crypto"
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
 * Whether to spend this model's output budget on the ANSWER rather than on a
 * native reasoning channel.
 *
 * OpenRouter bills reasoning tokens against the same `max_tokens` cap the reply
 * comes out of, and `stream.text` collects only the text channel. A judge call
 * is a deliberately tight budget — ClaudeUI's classifier runs its first stage at
 * `maxTokens: 64` with a `</block>` stop sequence — so a natively-reasoning
 * model spends the entire cap thinking and returns `finish_reason: "length"`
 * with EMPTY text. To a fail-closed caller that is an unparseable verdict, i.e.
 * the judge fails exactly on the hard inputs it exists for. Observed live on
 * `deepseek/deepseek-v4-flash-latest`; with `reasoning: { enabled: false }` the
 * same model reports ~0 reasoning tokens and answers in seven.
 *
 * Nothing is lost by turning it off: the judge prompts carry their own inline
 * `<thinking>` protocol, so the deliberation happens in the text channel — the
 * only channel this route reads, and the one the verdict must arrive in.
 *
 * `llmgateway` is paired with `openrouter` because it takes the same request
 * body shape; `ProviderTransform.smallOptions` already treats the two together
 * when it disables reasoning for OpenRouter's Google models
 * (provider/transform.ts).
 *
 * This is an OVERRIDE, not a default: `smallOptions` seeds the options from the
 * model's first variant, and `ProviderTransform.reasoningVariants` synthesizes
 * variants for every model whose models.dev entry lists `reasoning_options` —
 * for OpenRouter that first variant is `{ reasoning: { effort: … } }`
 * (`reasoningEffort`, transform.ts). So a `reasoning` key here is derived
 * registry metadata, not a user's judge preference, and deferring to it
 * reinstates exactly the empty-verdict failure (verified live: the deepseek
 * model above ships effort variants, and the key check silently kept its
 * reasoning on). A judge that must answer in the text channel has no
 * reasoning-on configuration to respect.
 *
 * The injected value is a constant, so it stays byte-identical call to call and
 * cannot disturb the stable-prefix requirement the handler's caching note
 * depends on.
 */
function disablesNativeReasoning(model: Provider.Model): boolean {
  return (model.providerID === "openrouter" || model.providerID === "llmgateway") && model.capabilities.reasoning
}

/**
 * Headers the built-in plugins add. Only openai's are reproduced: the Codex
 * endpoint expects the `originator`/User-Agent shape the codex CLI sends
 * (plugin/openai/codex.ts:549-557). Copilot's extra headers are version and
 * interaction hints its API tolerates the absence of, and its hook's
 * session-parts lookup has no judge-side equivalent.
 */
function providerHeaders(model: Provider.Model, cacheID: string): Record<string, string> {
  if (model.providerID !== "openai") return { "User-Agent": USER_AGENT }
  return {
    originator: "opencode",
    "User-Agent": `opencode/${InstallationVersion} (${os.platform()} ${os.release()}; ${os.arch()})`,
    // The codex hook sends the *session* id here (codex.ts:552). It is a
    // routing/affinity key, so a per-call random value would scatter otherwise
    // identical judge calls across cache shards. See {@link cacheID}.
    "session-id": cacheID,
  }
}

/**
 * Prompt-cache identity for this call, derived from the SYSTEM prompt ALONE.
 *
 * The session path keys caching off `sessionID`
 * (`ProviderTransform.options()`), which this route has no equivalent of. What
 * it does have is something better suited: the judge's system prompt is a large
 * (~24 KB for ClaudeUI's policy), byte-stable document reused across every call
 * of a session, while the `user` part is a fresh transcript each time. Hashing
 * the system prompt therefore names exactly the cacheable prefix — the same
 * policy lands on the same cache shard whoever calls it, and a changed policy
 * gets a different key instead of thrashing an existing entry.
 *
 * Truncated to 128 bits: this is a routing hint, not a security boundary, and
 * the prompt itself is re-sent and re-validated by the provider on every call.
 */
function cacheID(system: string): string {
  return `judge-${createHash("sha256").update(system).digest("hex").slice(0, 32)}`
}

/**
 * Where each provider family wants the prompt-cache key. Mirrors the
 * `setCacheKey` block in `ProviderTransform.options()`
 * (provider/transform.ts) — same providers, same option names; only the value
 * differs (system-prompt hash instead of session id, see {@link cacheID}).
 *
 * Absent from this list are the providers that take explicit `cache_control`
 * breakpoints instead — those are handled by {@link systemMessage} — and the
 * ones with no prompt cache at all.
 */
function cacheKeyOptions(model: Provider.Model, id: string): Record<string, string> {
  if (model.api.npm === "@ai-sdk/deepinfra" || model.api.npm === "@ai-sdk/cerebras") return { prompt_cache_key: id }
  if (
    model.api.npm === "@ai-sdk/openai" ||
    model.api.npm === "@ai-sdk/azure" ||
    model.api.npm === "@ai-sdk/xai" ||
    model.api.npm === "@ai-sdk/mistral" ||
    model.api.npm === "venice-ai-sdk-provider"
  ) {
    return { promptCacheKey: id }
  }
  if (model.providerID.startsWith("opencode") && model.api.id.includes("gpt-5")) return { promptCacheKey: id }
  return {}
}

/**
 * The judge's system prompt as a message, carrying a cache breakpoint when the
 * provider wants one.
 *
 * Only the SYSTEM part is marked, and deliberately so. `ProviderTransform`'s
 * `applyCaching` marks up to four breakpoints (two system messages, the last
 * two non-system messages) because a session replays a growing conversation
 * whose tail is stable between turns. A judge call is one system + one user
 * turn where the user part is a *different transcript every time*: a
 * breakpoint there could never hit, and on providers that cap breakpoints
 * (Alibaba: 4) it would spend one of them for nothing.
 *
 * The provider gate and the marker set come from `ProviderTransform` rather
 * than being copied, so a provider upstream adds to `applyCaching` starts
 * being cached here in the same commit.
 */
function systemMessage(system: string, model: Provider.Model, options: Record<string, unknown>): ModelMessage {
  if (!ProviderTransform.usesCacheMarkers(model, options)) return { role: "system", content: system }
  return { role: "system", content: system, providerOptions: ProviderTransform.cacheMarkers() }
}

/** `stream.usage`, narrowed to the fields worth reporting. Mirrors the shape
 *  `session/llm/ai-sdk.ts` normalizes to, so a caller reading both sees one
 *  vocabulary. `cacheReadInputTokens` is the whole point of P3 — it is the
 *  only in-band proof that the system prompt was served from cache. */
function usageOf(value: unknown): Record<string, number> | undefined {
  if (!value || typeof value !== "object") return undefined
  const item = value as {
    inputTokens?: number
    outputTokens?: number
    totalTokens?: number
    reasoningTokens?: number
    cachedInputTokens?: number
    inputTokenDetails?: { cacheReadTokens?: number; cacheWriteTokens?: number }
    outputTokenDetails?: { reasoningTokens?: number }
  }
  const result: Record<string, number> = {}
  for (const [key, value] of Object.entries({
    inputTokens: item.inputTokens,
    outputTokens: item.outputTokens,
    totalTokens: item.totalTokens,
    reasoningTokens: item.outputTokenDetails?.reasoningTokens ?? item.reasoningTokens,
    cacheReadInputTokens: item.inputTokenDetails?.cacheReadTokens ?? item.cachedInputTokens,
    cacheWriteInputTokens: item.inputTokenDetails?.cacheWriteTokens,
  })) {
    if (typeof value === "number") result[key] = value
  }
  return Object.keys(result).length === 0 ? undefined : result
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
 *
 * ## Prompt caching (ADR-037 P3)
 *
 * A judge call is a huge, stable system prompt plus a small, always-different
 * user turn. That is the ideal caching shape, and the route serves it two ways
 * depending on what the provider offers:
 *
 *  - **Explicit breakpoints** (Anthropic-style `cache_control`, plus Alibaba,
 *    OpenRouter, Bedrock, Copilot): one marker on the system message, none on
 *    the user turn — see {@link systemMessage}.
 *  - **Automatic prefix caching** (OpenAI Responses and friends): nothing to
 *    mark; the provider hashes the leading bytes of the request itself. All
 *    this route has to do is not defeat it, which imposes a real constraint:
 *    **everything ahead of the user turn must be byte-identical call to call.**
 *    It is — and each of these is load-bearing:
 *      * `options` comes from `ProviderTransform.smallOptions(model)`, which is
 *        a pure function of the model. The session path's
 *        `ProviderTransform.options()` — the one that injects `sessionID` as a
 *        cache key — is deliberately NOT used here.
 *      * `instructions` (the OAuth transport's system channel) is
 *        `payload.system` verbatim: no request id, timestamp, cwd or session id
 *        is spliced in anywhere on the system path.
 *      * `temperature` / `topP` / `topK` / `maxOutputTokens` derive from the
 *        model and the payload only.
 *      * the one value that *was* per-call random — the `session-id` header —
 *        is now {@link cacheID}, a hash of the system prompt.
 *    The corollary belongs to the CALLER: a system prompt that changes between
 *    calls (a clock, a counter, a re-ordered set) silently costs a full
 *    uncached prefix every time.
 *
 * `usage` on the response carries `cacheReadInputTokens` where the provider
 * reports it, so the caller can tell a cache hit from a hopeful one.
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
      const id = cacheID(payload.system)
      const options = { ...ProviderTransform.smallOptions(model), ...cacheKeyOptions(model, id) }
      if (disablesNativeReasoning(model)) options.reasoning = { enabled: false }
      if (isOpenaiOauth) options.instructions = payload.system

      const ceiling = ProviderTransform.maxOutputTokens(model)
      const maxOutputTokens = rejectsOutputCap(model)
        ? undefined
        : payload.maxTokens === undefined
          ? ceiling
          : Math.max(1, Math.min(Math.floor(payload.maxTokens), ceiling))
      const stopSequences = payload.stopSequences?.filter((x) => x.length > 0) ?? []
      // On the OAuth transport the system prompt travels as `instructions`, so
      // there is no system message to mark — and nothing to mark it with, since
      // that transport caches automatically.
      const messages: ModelMessage[] = [
        ...(isOpenaiOauth ? [] : [systemMessage(payload.system, model, options)]),
        { role: "user", content: payload.user },
      ]

      const result = yield* Effect.tryPromise({
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
            // `messages`, not `system` + `prompt`: a cache breakpoint is
            // attached to the system MESSAGE, and the string form has nowhere
            // to hang `providerOptions`.
            messages,
            ...(maxOutputTokens === undefined ? {} : { maxOutputTokens }),
            ...(stopSequences.length > 0 ? { stopSequences: [...stopSequences] } : {}),
            temperature: model.capabilities.temperature ? ProviderTransform.temperature(model) : undefined,
            topP: ProviderTransform.topP(model),
            topK: ProviderTransform.topK(model),
            providerOptions: ProviderTransform.providerOptions(model, options),
            headers: { ...model.headers, ...providerHeaders(model, id) },
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
          // Awaited after the text so a usage promise that never settles on a
          // failed stream cannot hide the real error.
          return { text: collected, usage: usageOf(await stream.usage) }
        },
        catch: (cause) => upstream(model.providerID, cause),
      })

      return result.usage === undefined ? { text: result.text } : { text: result.text, usage: result.usage }
    })

    return handlers.handle("completion", completion)
  }),
)
