# Algai Development Workflow

This project uses a "Central Hub" strategy with GitHub as the source of truth.

## Roles
- **GitHub**: The Source of Truth. Stores code, manages branches, and handles Pull Requests.
- **Google Jules**: Remote Senior Developer. Handles complex, asynchronous tasks (e.g., "Add login system"). Runs on Google Cloud.
- **Google Antigravity**: Local Super-Editor (IDE). Handles hands-on, iterative work (building, debugging).

## How to Work

### 1. The Setup
- **Do not** upload zip folders.
- **Do** sync with GitHub.

### 2. using Antigravity (Local)
1. Clone/Pull the latest `main` branch.
2. Open in Antigravity (VS Code).
3. Use the Agent Manager for real-time help.
4. **Commit & Push** changes to GitHub to save work.

### 3. Using Jules (Remote)
1. Go to [jules.google.com](https://jules.google.com).
2. Point Jules to this GitHub repo.
3. Assign a task (e.g., "Refactor the valuation engine").
4. Jules creates a Branch & PR.
5. Review and Merge the PR on GitHub.
6. `git push -u origin main`
Once that push succeeds, your Jules agent and Claude Code will be fully synced with your GitHub repo!

7. `git pull` locally to get Jules' changes.

## Claude Code Integration (Bridge)

You can use your Claude Opus subscription with Antigravity tools using the proxy bridge.

### Setup
1. **Initialize Proxy**: This is done once after installation.
   ```bash
   antigravity-claude-proxy accounts add
   ```
2. **Configure Claude Settings**:
   The setting `ANTHROPIC_BASE_URL` in `%USERPROFILE%\.claude\settings.json` must be set to `http://localhost:8080`.

### Daily Workflow
1. **Start the Bridge**:
   ```bash
   antigravity-claude-proxy start
   ```
2. **Run Claude**:
   In a new terminal, run `claude`. It will now route requests through Antigravity, utilizing your high-tier subscription via the IDE's enhanced environment.

## Summary of Commands
- `npm install -g @google/jules`: Install Jules CLI.
- `npm install -g antigravity-claude-proxy`: Install Claude Bridge.
- `git pull`: Update local code.
- `git push`: Send local changes to GitHub.
