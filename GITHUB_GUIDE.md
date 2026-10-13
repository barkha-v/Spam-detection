# Uploading this project to GitHub (Windows, VS Code or PyCharm)

These steps create a **new** repository, so nothing existing in your GitHub account is overwritten or deleted.

## 1. One-time Git setup (your own identity)
Install Git from https://git-scm.com, then:
```
git config --global user.name "Your Name"
git config --global user.email "your-verified-email@example.com"
git config --global user.name
git config --global user.email
```
Use an email that is added and verified in GitHub → Settings → Emails (or GitHub's private `no-reply` address shown there). Commits are attributed by this email. Contributor display depends on commit history and metadata, so don't expect anything specific beyond using your own identity.

## 2. Create the repository on GitHub
1. Sign in at https://github.com/barkha-v and click **New** (or https://github.com/new).
2. Repository name: `spam-email-detection`.
3. Add a short description. Choose Public or Private.
4. **Do NOT** tick "Add a README", ".gitignore" or "license" (the project already has them).
5. Click **Create repository**.

## 3. First push
Open a terminal in the project folder:
```
git init
git add .
git status
git commit -m "Initial commit: spam email detection project"
git branch -M main
git remote add origin https://github.com/barkha-v/spam-email-detection.git
git push -u origin main
```
GitHub will ask you to sign in (browser window or a Personal Access Token – never put tokens in project files).

Before committing, `git status` should NOT list `venv/`, `data/sms*`/`SMSSpamCollection`, or `models/*.joblib` – the `.gitignore` excludes them.

## 4. Later updates
```
git status
git add .
git commit -m "Describe what you changed"
git push
```

## 5. Checking your setup
| Command | Shows |
|---------|-------|
| `git status` | changed/untracked files and what's staged |
| `git remote -v` | the GitHub URL your repo pushes to (should be `.../barkha-v/spam-email-detection.git`) |
| `git branch` | local branches; the current one has `*` (should be `main`) |
| `git log --oneline` | commit history |

## 6. Common problems
- **`remote origin already exists`** → `git remote set-url origin https://github.com/barkha-v/spam-email-detection.git`
- **`failed to push ... rejected`** (GitHub repo has files you don't have locally) → `git pull origin main --allow-unrelated-histories`, resolve any conflicts, then push. Don't use `--force`.
- **Wrong author on a commit** → fix `git config` (step 1) and make new commits.

## 7. In VS Code / PyCharm
Both have a Git panel (VS Code: Source Control icon; PyCharm: *Git* menu) that performs the same add/commit/push actions. The terminal commands above work in both.

## 8. After uploading
Add screenshots (see README) and a repository description/topics such as `machine-learning`, `nlp`, `naive-bayes`, `streamlit`.
