---
name: skill-creator
description: Create, refine, or organize Choreo-specific Claude skills. Use when the user asks to add a new skill, improve trigger descriptions, document a new workflow, or prepare skills for open-source sharing. Invoke directly with `/skill-creator`.
argument-hint: [skill name or what to create]
disable-model-invocation: true
---

# Skill Creator

Create or improve a Choreo project skill from the request: `$ARGUMENTS`.

This skill helps agents self-promote and extend the project's skill ecosystem. Use it to codify common workflows, package new compiler features as discoverable skills, or maintain consistency across the skill library.

Follow this workflow:

1. **Understand the domain**: Read existing Choreo skills under `${CLAUDE_SKILL_DIR}/../` to match their structure, tone, and naming conventions. This project uses:
   - Short, action-oriented skill names (e.g., `build-and-install`, `compile-and-test`, `develop-feature`)
   - Descriptions that explain both what the skill does and when to invoke it
   - Supporting files for detailed reference

2. **Classify the skill type**:
   - **Reference content** — conventions, patterns, domain knowledge (load automatically)
   - **Task/workflow content** — build, test, debug, profile actions (often manual-only)
   - **Hybrid** — reference + task (choose based on primary use)

3. **Choose frontmatter deliberately**:
   | Field | When to use |
   |-------|-------------|
   | `description` | Always. Say what it does AND when to use it. |
   | `disable-model-invocation: true` | For side-effectful workflows (build, deploy, commit). |
   | `user-invocable: false` | For background knowledge only. |
   | `argument-hint` | When slash-command expects arguments. |
   | `allowed-tools` | Only when tighter tool scope improves safety. |

4. **Structure the skill**:
   - Keep `SKILL.md` under 500 lines; put the most important instructions first.
   - Use supporting files for templates, detailed references, examples, or scripts.
   - Make it self-promoting: mention where it triggers, reference related skills, and describe how it fits into Choreo workflows.

5. **Mirror if needed**: If the repo maintains parallel skill trees (e.g., `.claude/skills/` for open-source, `.github/skills/` for full internal), create the skill in both unless the user specifies otherwise.

6. **Validate**:
   - Name uses lowercase letters, numbers, and hyphens only.
   - Description triggers correctly on realistic user requests.
   - Instructions are concrete and executable.
   - No private backend details leak into the public copy.

Use the templates in [template.md](template.md) and the checklist in [best-practices.md](best-practices.md).

**When finished, report**:
- Which files were created or updated
- The invocation model you chose and why
- Any follow-up suggestions for discoverability
