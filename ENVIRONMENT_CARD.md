# Environment card

`founder_decision_env` is an OpenEnv v0.4.1-compatible FastAPI/WebSocket environment.

- Action: one of `interview_users`, `build_feature`, `sell_pilot`, `fundraise`,
  `change_price`, `abstain`, plus explicit costs, source locator, rationale, and state claim.
- Observation: visible synthetic state, allowed actions/locators, component rewards,
  failure codes, and termination flags.
- State: episode/case/seed, four-week counter, budget, founder hours, users, interviews,
  pipeline, MRR, price, and frozen model version.
- Reset: deterministic case selection and deterministic episode ID from seed.
- Step: action-dependent frozen transition; hard failures consume a bounded step but cause no
  business-state side effect.
- End: `terminated` on resource exhaustion; otherwise `truncated` after four steps.

OpenEnv's own local/runtime validator is necessary but does not test replay, persistent state,
reward components, or termination semantics. Project tests cover those house contracts.
