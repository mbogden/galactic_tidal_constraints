# galactic_tidal_constraints

Simulates galaxy collisions with SPAM and compares them to observed and simulated galaxies, to study
what tidal features can tell us about how galaxies interact. Formerly `galaxyJSPAM`.

> **Status: being rewritten.** This `modern` branch is a from-scratch rewrite as an installable,
> reproducible package, and most of it doesn't exist yet. The working research code is on the
> [`working_research`](https://github.com/mbogden/galactic_tidal_constraints/tree/working_research)
> branch. Decisions behind the rewrite are in [PLAN.md](PLAN.md).

## SPAM
Collisions are simulated with SPAM, by John Wallin and collaborators, through a modified fork of the
original code:

- Original: [jfwallin/JSPAM](https://github.com/jfwallin/JSPAM)
- Fork used here: [mbogden/JSPAM](https://github.com/mbogden/JSPAM)

## Try it
[docs/running_spam.md](docs/running_spam.md) explains how to compile and run SPAM. With Docker and
git installed, and JSPAM cloned next to this repo, this checks that the original code runs correctly
on your machine (use bash; Git Bash on Windows):
```bash
bash examples/spam/check_original_jspam.sh
```
This repo doesn't call SPAM from its own code yet.

## License
Licensed under the Academic Free License version 3.0. See [LICENSE](LICENSE). This project's use of
SPAM is modified from the original, as noted in the fork's README.
