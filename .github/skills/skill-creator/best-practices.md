# Skill Authoring Best Practices

This file captures the practical guidance for creating Claude skills that trigger well and stay maintainable.

## Core principles

- Write a clear `description` that includes both what the skill does and when to use it.
- Keep `SKILL.md` short; move bulky details into supporting files.
- Separate reference skills from task skills.
- Use manual-only invocation for workflows with side effects.
- Prefer explicit, stepwise instructions over vague guidance.
- Mirror existing repo conventions so the skill feels native to the project.

## Description writing

A good description usually contains:

- the action,
- the context,
- trigger phrases a user might naturally say.

Example pattern:

`Create or refine project skills for Claude Code. Use when adding a new skill, improving trigger descriptions, or packaging reusable project workflows.`

Avoid descriptions that are too short, too generic, or overloaded with unrelated keywords.

## Frontmatter choices

- `disable-model-invocation: true`
  - Use for deployment, commit, publishing, migrations, or other side-effectful flows.
- `user-invocable: false`
  - Use for passive knowledge that should only load automatically.
- `argument-hint`
  - Add when arguments are expected, especially for slash-command use.
- `allowed-tools`
  - Add only when a bounded tool policy helps safety or behavior.

## Structure of a good skill

Keep `SKILL.md` focused on:

1. what the skill is for,
2. the execution workflow,
3. quality checks,
4. references to supporting files.

Useful supporting files:

- `template.md` for a starter scaffold,
- `examples.md` for good samples,
- `reference.md` for detailed domain notes,
- `scripts/` for helper automation.

## Discoverability and self-promotion

To help Claude discover and reuse project skills:

- give each skill a distinctive description,
- avoid overlapping descriptions that compete too much,
- reference sibling skills when there is a natural handoff,
- store project skills in `.claude/skills/`,
- keep mirrored copies organized if the repo also maintains `.github/skills/`.

## Open-source hygiene

When a public and private skill tree both exist:

- keep secret backend details out of the public tree,
- scrub proprietary target names and internal infra references,
- preserve only the safe, open workflow guidance.

## Review checklist

- Is the name stable and slash-command friendly?
- Will the description trigger on realistic user requests?
- Is the invocation mode correct?
- Are the steps concrete enough to execute?
- Should any detailed content move into a supporting file?
- Does the public copy avoid private details?
