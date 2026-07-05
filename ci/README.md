# CI (pending activation)

`gates.yml` runs the full gate suite on CPU (verified: 15/15 in ~2.5 min).
GitHub only executes workflows under `.github/workflows/`. Activate with:

    mkdir -p .github/workflows && git mv ci/gates.yml .github/workflows/
    git commit -m "activate gate CI" && git push
