import { Auth } from "@/auth"
import { Provider } from "@/provider/provider"
import { ProviderTransform } from "@/provider/transform"
import { InstallationVersion } from "@opencode-ai/core/installation/version"
import { generateText } from "ai"
import { Effect } from "effect"
import { HttpApiBuilder } from "effect/unstable/httpapi"
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
 * `maxTokens` and `stopSequences` are honoured for real (the session prompt API
 * exposes neither). `maxTokens` is clamped to the model's own output ceiling.
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
      const maxOutputTokens =
        payload.maxTokens === undefined ? ceiling : Math.max(1, Math.min(Math.floor(payload.maxTokens), ceiling))
      const stopSequences = payload.stopSequences?.filter((x) => x.length > 0) ?? []

      const result = yield* Effect.tryPromise({
        try: (signal) =>
          generateText({
            model: language,
            ...(isOpenaiOauth ? {} : { system: payload.system }),
            prompt: payload.user,
            maxOutputTokens,
            ...(stopSequences.length > 0 ? { stopSequences: [...stopSequences] } : {}),
            temperature: model.capabilities.temperature ? ProviderTransform.temperature(model) : undefined,
            topP: ProviderTransform.topP(model),
            topK: ProviderTransform.topK(model),
            providerOptions: ProviderTransform.providerOptions(model, options),
            headers: { ...model.headers, "User-Agent": USER_AGENT },
            maxRetries: 0,
            abortSignal: signal,
          }),
        catch: (cause) =>
          new UpstreamError({
            message: cause instanceof Error ? cause.message : String(cause),
            service: model.providerID,
          }),
      })

      return { text: result.text }
    })

    return handlers.handle("completion", completion)
  }),
)
