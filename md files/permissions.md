# Claude Code Permissions

Review and adjust these lists, then copy the JSON at the bottom into `.claude/settings.local.json`.

---

## ALLOW — Pre-approved, never prompt

These are safe, read-only, or non-destructive. Claude can run them freely.

### Git — read-only
| Command pattern | Why |
|---|---|
| `Bash(git status)` | Read-only, used constantly |
| `Bash(git diff*)` | Read-only, inspecting changes |
| `Bash(git log*)` | Read-only, inspecting history |
| `Bash(git branch*)` | Read-only, listing branches |
| `Bash(git show*)` | Read-only, inspecting commits |
| `Bash(git remote*)` | Read-only, checking remotes |

### Environment inspection
| Command pattern | Why |
|---|---|
| `Bash(python --version*)` | Checking Python version |
| `Bash(pip show*)` | Checking if a package is installed |
| `Bash(pip list*)` | Listing installed packages |
| `Bash(pip freeze*)` | Exporting package versions |
| `Bash(python -c "import*)` | Quick inline import/version checks |
| `Bash(nvidia-smi*)` | Checking GPU availability |

### File system — non-destructive
| Command pattern | Why |
|---|---|
| `Bash(mkdir -p*)` | Creating directories, never overwrites |
| `Bash(ls*)` | Listing files |
| `Bash(cat*)` | Reading files (use Read tool instead when possible) |

### Verification
| Command pattern | Why |
|---|---|
| `Bash(python -m pytest*)` | Running tests |
| `Bash(python setup.py*)` | Package setup checks |

### Claude / plugin management
| Command pattern | Why |
|---|---|
| `Bash(claude plugin list*)` | Listing installed plugins |
| `Bash(claude plugin install*)` | Installing plugins |
| `Bash(claude plugin marketplace*)` | Managing marketplaces |
| `Bash(claude mcp*)` | Managing MCP servers |

---

## ASK — Prompt before running

These are write operations, expensive, or potentially destructive.

### Git — write operations
*User policy: never commit or push without explicit approval.*
| Command pattern | Reason to ask |
|---|---|
| `Bash(git add*)` | Staging files before commit |
| `Bash(git commit*)` | Creating commits |
| `Bash(git push*)` | Pushing to remote — irreversible |
| `Bash(git checkout*)` | Can discard local changes |
| `Bash(git reset*)` | Can lose uncommitted work |
| `Bash(git stash*)` | Hides changes, easy to forget |
| `Bash(git merge*)` | Modifies branch history |
| `Bash(git rebase*)` | Rewrites history |
| `Bash(git restore*)` | Discards changes |
| `Bash(git clean*)` | Deletes untracked files |

### Package installation
| Command pattern | Reason to ask |
|---|---|
| `Bash(pip install*)` | Modifies Python environment |
| `Bash(pip uninstall*)` | Removes packages |

### Training scripts — expensive (Colab credits)
| Command pattern | Reason to ask |
|---|---|
| `Bash(python scripts/run_sft.py*)` | Runs SFT training, hours of compute |
| `Bash(python scripts/run_grpo.py*)` | Runs GRPO training, hours of compute |
| `Bash(python scripts/run_eval.py*)` | Runs full evaluation, significant compute |
| `Bash(python notebooks/*)` | Running notebooks directly |

### Destructive file operations
| Command pattern | Reason to ask |
|---|---|
| `Bash(rm*)` | Deletes files, hard to undo |
| `Bash(mv*)` | Moves/renames, can overwrite |
| `Bash(cp -r*)` | Bulk copy, could fill disk |

---

## settings.local.json

This is the file to update: `.claude/settings.local.json`

```json
{
  "permissions": {
    "allow": [
      "Bash(git status)",
      "Bash(git diff*)",
      "Bash(git log*)",
      "Bash(git branch*)",
      "Bash(git show*)",
      "Bash(git remote*)",
      "Bash(python --version*)",
      "Bash(pip show*)",
      "Bash(pip list*)",
      "Bash(pip freeze*)",
      "Bash(python -c \"import*)",
      "Bash(nvidia-smi*)",
      "Bash(mkdir -p*)",
      "Bash(ls*)",
      "Bash(cat*)",
      "Bash(python -m pytest*)",
      "Bash(python setup.py*)",
      "Bash(claude plugin list*)",
      "Bash(claude plugin install*)",
      "Bash(claude plugin marketplace*)",
      "Bash(claude mcp*)"
    ]
  }
}
```

> The "ask" commands don't need to be listed — anything not in `allow` is automatically prompted. Only list something explicitly in `ask` if you want to override a parent-level `allow` rule.

---

## Notes

- The current `settings.local.json` already has some plugin install commands in `allow` from this session — those are covered by the broader `Bash(claude plugin install*)` pattern above.
- `settings.local.json` is project-scoped and not committed to git (it's in `.claude/`). User-level settings live in `C:\Users\the4r\.claude\settings.json`.
- If a command isn't in `allow`, Claude will pause and show you the exact command before running it.
