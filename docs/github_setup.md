# Publish to your GitHub portfolio

Suggested repository name: `football-match-forecasting`

Suggested description: `Interpretable football forecasting with opponent-adjusted npxG ratings, Poisson/Dixon-Coles probabilities, chronological evaluation, and reproducible tests.`

1. In `juancamelo234-svg`, create `football-match-forecasting`. Add an initial README so GitHub creates the default branch. Use public visibility for a recruiter-accessible portfolio, or private while reviewing.
2. Upload the **contents** of this project folder at the repository root, including `.github/workflows/tests.yml` and `.gitignore`. Avoid an extra outer folder. Do not upload the ZIP file itself.
3. Alternatively, initialize this folder with Git, add it to your repository, and push with your own authenticated GitHub account.
4. Inspect the rendered README, open the method and data definitions, and check the Actions run. CI has to execute on GitHub before a hosted test pass can be claimed.
5. Make the repository public when ready and pin it on your profile. Add its real URL to your resume project title.

Suggested topics: `python`, `sports-analytics`, `football`, `statistical-modeling`, `probabilistic-forecasting`, `dixon-coles`.

Do not advertise live deployment, XGBoost retraining in this compact release, or market outperformance. Use the README's precise evidence and scope.

## Terminal upload

After creating the GitHub repository with its initial README, clone it and copy this package's contents into the clone. This preserves the remote history. Then run:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/verify_research.py
git add README.md pyproject.toml src tests docs research scripts .github .gitignore
git commit -m "Add forecasting core, archived metric reproduction, and tests"
git push origin main
```

Do not add `data/private/`, the original research ZIP, credentials, or local outputs. If the connected GitHub integration cannot see the repository, grant it access to this specific repository before requesting an upload.
