import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { PermissionV1 } from "@opencode-ai/core/v1/permission"
import { SessionID } from "@opencode-ai/schema/session-id"
import { describe, expect } from "bun:test"
import { Effect, Fiber } from "effect"
import { Permission } from "../../src/permission"
import { testEffect } from "../lib/effect"

const it = testEffect(LayerNode.compile(Permission.node))

const DENY_ALL: PermissionV1.Ruleset = [{ permission: "*", pattern: "*", action: "deny" }]

const NORMAL = SessionID.make("ses_normal000000000000000000")
const SEALED = SessionID.make("ses_sealed000000000000000000")
const APPROVER = SessionID.make("ses_approver00000000000000000")

/**
 * Make `bash: git *` an instance-global "always" approval, exactly the way a
 * real user does: a session asks, and the human answers "always".
 */
const alwaysApproveGit = Effect.fn("alwaysApproveGit")(function* () {
  const perm = yield* Permission.Service
  const id = PermissionV1.ID.ascending()
  const asking = yield* Effect.forkScoped(
    perm.ask({
      id,
      sessionID: APPROVER,
      permission: "bash",
      patterns: ["git *"],
      metadata: {},
      always: ["git *"],
      ruleset: [],
    }),
  )
  // Let the ask register before replying to it.
  yield* Effect.sleep("10 millis")
  yield* perm.reply({ requestID: id, reply: "always" })
  yield* Fiber.join(asking)
})

/** Would `bash: git status` be allowed straight through for this session? */
const tryGit = Effect.fn("tryGit")(function* (sessionID: SessionID, ruleset: PermissionV1.Ruleset) {
  const perm = yield* Permission.Service
  return yield* perm
    .ask({
      sessionID,
      permission: "bash",
      patterns: ["git status"],
      metadata: {},
      always: [],
      ruleset,
    })
    .pipe(
      Effect.map(() => "allowed" as const),
      Effect.catchTag("PermissionDeniedError", () => Effect.succeed("denied" as const)),
      // An unresolved ask would block forever; a timeout means "went to the human".
      Effect.timeoutOrElse({ duration: "300 millis", orElse: () => Effect.succeed("asked" as const) }),
    )
})

describe("Permission hermetic sessions", () => {
  it.instance("an instance-global always-approval pierces a normal session's deny-all", () =>
    Effect.gen(function* () {
      yield* alwaysApproveGit()
      // This is upstream behaviour, not a bug we introduced: `approved` is
      // appended AFTER the session ruleset and `findLast` wins.
      expect(yield* tryGit(NORMAL, DENY_ALL)).toBe("allowed")
    }),
  )

  it.instance("a sealed session's deny-all holds against the same approval", () =>
    Effect.gen(function* () {
      const perm = yield* Permission.Service
      yield* alwaysApproveGit()
      yield* perm.seal({ sessionID: SEALED, hermetic: true })
      expect(yield* tryGit(SEALED, DENY_ALL)).toBe("denied")
    }),
  )

  it.instance("sealing one session does not change any other session", () =>
    Effect.gen(function* () {
      const perm = yield* Permission.Service
      yield* alwaysApproveGit()
      yield* perm.seal({ sessionID: SEALED, hermetic: true })
      expect(yield* tryGit(SEALED, DENY_ALL)).toBe("denied")
      expect(yield* tryGit(NORMAL, DENY_ALL)).toBe("allowed")
    }),
  )

  it.instance("unsealing restores the default evaluation", () =>
    Effect.gen(function* () {
      const perm = yield* Permission.Service
      yield* alwaysApproveGit()
      yield* perm.seal({ sessionID: SEALED, hermetic: true })
      expect(yield* tryGit(SEALED, DENY_ALL)).toBe("denied")
      yield* perm.seal({ sessionID: SEALED, hermetic: false })
      expect(yield* tryGit(SEALED, DENY_ALL)).toBe("allowed")
    }),
  )

  it.instance("a sealed session still honours its OWN allow rules", () =>
    Effect.gen(function* () {
      const perm = yield* Permission.Service
      yield* perm.seal({ sessionID: SEALED, hermetic: true })
      expect(yield* tryGit(SEALED, [{ permission: "bash", pattern: "git *", action: "allow" }])).toBe("allowed")
    }),
  )

  it.instance("an 'always' answered inside a sealed session does not widen the instance", () =>
    Effect.gen(function* () {
      const perm = yield* Permission.Service
      yield* perm.seal({ sessionID: SEALED, hermetic: true })

      const id = PermissionV1.ID.ascending()
      const asking = yield* Effect.forkScoped(
        perm.ask({
          id,
          sessionID: SEALED,
          permission: "bash",
          patterns: ["git status"],
          metadata: {},
          always: ["git *"],
          ruleset: [],
        }),
      )
      yield* Effect.sleep("10 millis")
      yield* perm.reply({ requestID: id, reply: "always" })
      yield* Fiber.join(asking)

      // A sealed session is a sink, not a source: an unrelated session must
      // still have to ask.
      expect(yield* tryGit(NORMAL, [])).toBe("asked")
    }),
  )
})
